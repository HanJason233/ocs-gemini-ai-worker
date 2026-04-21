const DEFAULT_MODEL = "gemini-2.5-flash";
const DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta";
const AUTO_MODEL_MARKERS = new Set(["", "auto", "rotation", "rotate", "all", "轮询"]);

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Requested-With",
};

const rotationState = {
  key: 0,
  model: 0,
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(request.url);

    if (url.pathname === "/" || url.pathname === "/health") {
      return withCors(
        jsonResponse({
          code: 1,
          name: "ocs-gemini-answer-bank",
          endpoints: ["/api/answer", "/proxy/gemini/*"],
          defaultModel: getDefaultModel(env),
          modelPoolSize: getEnvModelPool(env).length,
          keyPoolSize: getApiKeyPool(env).length,
        }),
      );
    }

    if (url.pathname === "/api/answer") {
      return withCors(await handleAnswer(request, url, env));
    }

    if (url.pathname.startsWith("/proxy/gemini/")) {
      return withCors(await handleGeminiProxy(request, url, env));
    }

    return withCors(jsonResponse({ code: 0, msg: "Not Found" }, 404));
  },
};

async function handleAnswer(request, url, env) {
  const payload = await parsePayload(request, url);

  const title = pickString(payload, ["title", "question", "q"]).trim();
  const options = pickString(payload, ["options", "option", "choices"]).trim();
  const type = normalizeType(pickString(payload, ["type", "questionType", "question_type"]));
  const requestedModel = pickString(payload, ["model", "models"]).trim();
  const debug = pickString(payload, ["debug"]) === "1";

  if (!title) {
    return jsonResponse({ code: 0, msg: "Missing required field: title" }, 400);
  }

  const keyPool = getApiKeyPool(env);
  if (keyPool.length === 0) {
    return jsonResponse(
      {
        code: 0,
        msg: "No Google AI keys configured. Set secret GOOGLE_AI_KEYS (multi) or GOOGLE_AI_KEY (single).",
      },
      500,
    );
  }

  const modelPool = getRequestedModelPool(requestedModel, env);
  if (modelPool.length === 0) {
    return jsonResponse({ code: 0, msg: "No Gemini models configured." }, 500);
  }

  const prompt = buildPrompt({ title, options, type });
  const upstream = await callGeminiWithRotation({
    baseUrl: getBaseUrl(env),
    keyPool,
    modelPool,
    prompt,
  });

  if (!upstream.ok) {
    return jsonResponse(
      {
        code: 0,
        msg: upstream.error || "Gemini request failed",
        status: upstream.status || 502,
        attempts: debug ? upstream.attempts : undefined,
      },
      502,
    );
  }

  const answer = normalizeAnswer(upstream.text, type);
  if (!answer || answer.toUpperCase() === "UNKNOWN") {
    return jsonResponse({
      code: 0,
      msg: "No reliable answer found",
      question: title,
      model: upstream.model,
      raw: debug ? upstream.text : undefined,
    });
  }

  return jsonResponse({
    code: 1,
    question: title,
    answer,
    model: upstream.model,
    raw: debug ? upstream.text : undefined,
  });
}

async function handleGeminiProxy(request, url, env) {
  const keyPool = getApiKeyPool(env);
  if (keyPool.length === 0) {
    return jsonResponse(
      { code: 0, msg: "No Google AI keys configured. Set secret GOOGLE_AI_KEYS or GOOGLE_AI_KEY." },
      500,
    );
  }

  const path = url.pathname.replace(/^\/proxy\/gemini\/?/, "");
  if (!path) {
    return jsonResponse({ code: 0, msg: "Missing proxy path, for example /proxy/gemini/models" }, 400);
  }

  const baseUrl = getBaseUrl(env);
  const method = request.method.toUpperCase();
  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.delete("content-length");

  const filteredSearch = [];
  for (const [k, v] of url.searchParams.entries()) {
    if (k.toLowerCase() !== "key") {
      filteredSearch.push([k, v]);
    }
  }

  const bodyBuffer = method === "GET" || method === "HEAD" ? undefined : await request.arrayBuffer();
  const keyOrder = rotateByCursor(keyPool, "key");
  let lastResponse = null;

  for (let i = 0; i < keyOrder.length; i += 1) {
    const key = keyOrder[i];
    const target = new URL(`${baseUrl}/${path}`);
    for (const [k, v] of filteredSearch) {
      target.searchParams.append(k, v);
    }
    target.searchParams.set("key", key);

    const init = {
      method,
      headers,
      redirect: "follow",
      body: bodyBuffer ? bodyBuffer.slice(0) : undefined,
    };

    const upstream = await fetch(target, init);
    if (!shouldRetryKey(upstream.status) || i === keyOrder.length - 1) {
      return proxyResponse(upstream);
    }

    // Consume body before retrying next key.
    await upstream.arrayBuffer();
    lastResponse = upstream;
  }

  return jsonResponse(
    {
      code: 0,
      msg: "Gemini proxy failed after rotating all keys",
      status: lastResponse ? lastResponse.status : 502,
    },
    502,
  );
}

async function callGeminiWithRotation(input) {
  const modelOrder = rotateByCursor(input.modelPool, "model");
  const keyOrder = rotateByCursor(input.keyPool, "key");
  const attempts = [];

  for (const model of modelOrder) {
    for (const key of keyOrder) {
      const result = await callGeminiSingle({
        apiKey: key,
        baseUrl: input.baseUrl,
        model,
        prompt: input.prompt,
      });

      if (result.ok) {
        return { ok: true, text: result.text, status: result.status, model };
      }

      const errorType = classifyFailure(result.status, result.error);
      attempts.push({
        model,
        key: maskKey(key),
        status: result.status,
        error: result.error || "unknown error",
        type: errorType,
      });

      if (errorType === "model") {
        break;
      }
    }
  }

  const last = attempts[attempts.length - 1];
  return {
    ok: false,
    status: last ? last.status : 502,
    error: last ? `All key/model attempts failed: ${last.error}` : "No attempt executed",
    attempts,
  };
}

async function callGeminiSingle(input) {
  const endpoint = `${input.baseUrl}/models/${encodeURIComponent(input.model)}:generateContent?key=${encodeURIComponent(
    input.apiKey,
  )}`;

  const body = {
    contents: [{ role: "user", parts: [{ text: input.prompt }] }],
    generationConfig: {
      temperature: 0.1,
      topP: 0.9,
    },
  };

  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

  const rawText = await response.text();
  if (!response.ok) {
    return {
      ok: false,
      status: response.status,
      text: "",
      error: extractError(rawText) || `${response.status} ${response.statusText}`,
    };
  }

  const parsed = safeJsonParse(rawText);
  const text = extractGeminiText(parsed) || rawText;

  return { ok: true, status: response.status, text };
}

function buildPrompt(input) {
  return [
    "You are an online course answer assistant.",
    "Return only the final answer. Do not explain reasoning.",
    `Question type: ${input.type}`,
    `Question: ${input.title}`,
    input.options ? `Options:\n${input.options}` : "Options: none",
    "Output rules:",
    "1) single: one option letter (for example A) or short answer text.",
    "2) multiple: letters only, joined by #, for example A#C#D.",
    "3) judgement: only return Correct or Wrong.",
    "4) completion: return short answer text.",
    "If uncertain, return UNKNOWN only.",
  ].join("\n");
}

function normalizeAnswer(raw, type) {
  if (!raw) {
    return null;
  }

  const text = stripCodeFence(raw).trim();
  if (!text) {
    return null;
  }

  let candidate = text;
  const maybeJson = safeJsonParse(candidate);
  if (typeof maybeJson === "object" && maybeJson !== null) {
    const jsonAnswer = pickString(maybeJson, ["answer", "result", "output"]).trim();
    if (jsonAnswer) {
      candidate = jsonAnswer;
    }
  }

  candidate = candidate
    .replace(/^answer[:：]\s*/i, "")
    .replace(/^final\s*answer[:：]\s*/i, "")
    .trim();

  const firstLine = candidate
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find(Boolean);
  if (firstLine) {
    candidate = firstLine;
  }

  if (/^unknown$/i.test(candidate)) {
    return "UNKNOWN";
  }

  if (type === "judgement") {
    if (/(correct|true|yes)/i.test(candidate) && !/(wrong|false|no)/i.test(candidate)) {
      return "Correct";
    }
    if (/(wrong|false|no)/i.test(candidate)) {
      return "Wrong";
    }
  }

  if (type === "multiple") {
    if (candidate.includes("#")) {
      return candidate
        .split("#")
        .map((v) => v.trim().toUpperCase())
        .filter(Boolean)
        .join("#");
    }
    const letters = Array.from(new Set(candidate.toUpperCase().match(/[A-H]/g) || []));
    if (letters.length > 0) {
      return letters.join("#");
    }
  }

  return candidate.trim() || null;
}

async function parsePayload(request, url) {
  if (request.method.toUpperCase() === "GET") {
    return Object.fromEntries(url.searchParams.entries());
  }

  const contentType = request.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const parsed = safeJsonParse(await request.text());
    if (typeof parsed === "object" && parsed !== null) {
      return parsed;
    }
    return {};
  }

  if (contentType.includes("application/x-www-form-urlencoded") || contentType.includes("multipart/form-data")) {
    const form = await request.formData();
    const out = {};
    for (const [k, v] of form.entries()) {
      out[k] = typeof v === "string" ? v : v.name;
    }
    return out;
  }

  const raw = await request.text();
  if (!raw.trim()) {
    return {};
  }

  const parsed = safeJsonParse(raw);
  if (typeof parsed === "object" && parsed !== null) {
    return parsed;
  }
  return Object.fromEntries(new URLSearchParams(raw).entries());
}

function getRequestedModelPool(modelInput, env) {
  const raw = String(modelInput || "").trim();
  if (!AUTO_MODEL_MARKERS.has(raw.toLowerCase())) {
    const fromRequest = parseFlexibleList(raw).map(normalizeModelName).filter(Boolean);
    if (fromRequest.length > 0) {
      return uniq(fromRequest);
    }
  }
  return getEnvModelPool(env);
}

function getEnvModelPool(env) {
  const configured = parseFlexibleList(env.GEMINI_MODELS).map(normalizeModelName).filter(Boolean);
  if (configured.length > 0) {
    return uniq(configured);
  }

  const fallback = normalizeModelName(env.DEFAULT_MODEL || DEFAULT_MODEL);
  return fallback ? [fallback] : [DEFAULT_MODEL];
}

function getDefaultModel(env) {
  const modelPool = getEnvModelPool(env);
  return modelPool[0] || DEFAULT_MODEL;
}

function getApiKeyPool(env) {
  const keys = [
    ...parseFlexibleList(env.GOOGLE_AI_KEYS),
    ...parseFlexibleList(env.GOOGLE_AI_KEY),
  ]
    .map((item) => item.trim())
    .filter(Boolean)
    .filter((item) => item !== "REPLACE_WITH_YOUR_GOOGLE_AI_KEY");

  return uniq(keys);
}

function getBaseUrl(env) {
  return (env.GEMINI_BASE_URL || "").trim() || DEFAULT_GEMINI_BASE_URL;
}

function parseFlexibleList(input) {
  const raw = String(input || "").trim();
  if (!raw) {
    return [];
  }

  if (raw.startsWith("[")) {
    const parsed = safeJsonParse(raw);
    if (Array.isArray(parsed)) {
      return parsed.map((item) => String(item).trim()).filter(Boolean);
    }
  }

  return raw
    .split(/[\n,;|]+/g)
    .map((item) => item.trim())
    .filter(Boolean);
}

function normalizeModelName(model) {
  const normalized = String(model || "").trim().replace(/^models\//i, "").toLowerCase();
  const aliasMap = {
    "gemini2.5flash": "gemini-2.5-flash",
    "gemini2.5pro": "gemini-2.5-pro",
  };
  return aliasMap[normalized] || normalized;
}

function rotateByCursor(list, cursorName) {
  if (!Array.isArray(list) || list.length === 0) {
    return [];
  }
  const start = rotationState[cursorName] % list.length;
  rotationState[cursorName] = (start + 1) % list.length;
  return list.slice(start).concat(list.slice(0, start));
}

function classifyFailure(status, error) {
  const message = String(error || "").toLowerCase();
  if (status === 404) {
    return "model";
  }
  if (status === 400 && /(model|not found|unsupported|invalid model|does not exist)/i.test(message)) {
    return "model";
  }
  if (status === 401 || status === 403 || status === 429) {
    return "key";
  }
  if (status >= 500) {
    return "transient";
  }
  return "unknown";
}

function shouldRetryKey(status) {
  return status === 401 || status === 403 || status === 429;
}

function maskKey(key) {
  const value = String(key || "");
  if (value.length <= 8) {
    return "***";
  }
  return `${value.slice(0, 4)}...${value.slice(-4)}`;
}

function proxyResponse(upstream) {
  const responseHeaders = new Headers(upstream.headers);
  responseHeaders.delete("content-encoding");
  responseHeaders.delete("transfer-encoding");
  responseHeaders.delete("connection");

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders,
  });
}

function normalizeType(typeInput) {
  const value = String(typeInput || "").trim().toLowerCase();
  if (value === "single" || value === "multiple" || value === "judgement" || value === "completion") {
    return value;
  }
  return "single";
}

function extractGeminiText(parsed) {
  if (!parsed || typeof parsed !== "object") {
    return "";
  }

  const firstCandidate = parsed.candidates && parsed.candidates[0];
  const parts = (firstCandidate && firstCandidate.content && firstCandidate.content.parts) || [];
  return parts
    .map((part) => (typeof part.text === "string" ? part.text : ""))
    .filter(Boolean)
    .join("\n")
    .trim();
}

function extractError(rawText) {
  const parsed = safeJsonParse(rawText);
  if (parsed && typeof parsed === "object") {
    const message = pickString(parsed, ["message", "error_description"]).trim();
    if (message) {
      return message;
    }
    const error = parsed.error;
    if (error && typeof error === "object") {
      const nested = pickString(error, ["message", "status"]).trim();
      if (nested) {
        return nested;
      }
    }
  }
  return String(rawText || "").slice(0, 180).trim();
}

function stripCodeFence(value) {
  return String(value || "").replace(/^```[a-zA-Z0-9]*\s*/u, "").replace(/\s*```$/u, "");
}

function pickString(record, keys) {
  for (const key of keys) {
    const value = record && record[key];
    if (value === undefined || value === null) {
      continue;
    }
    return String(value);
  }
  return "";
}

function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function uniq(list) {
  return Array.from(new Set(list));
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
    },
  });
}

function withCors(response) {
  const headers = new Headers(response.headers);
  for (const [key, value] of Object.entries(CORS_HEADERS)) {
    headers.set(key, value);
  }
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}
