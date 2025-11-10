// -------------------- Required modules --------------------
import express from "express";
import fs from "fs";
import path from "path";
import natural from "natural";
import axios from "axios";
import translate from "translate-google";
import { fileURLToPath } from "url";
import { OpenAI } from "openai";

// -------------------- Setup --------------------
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3001;

// -------------------- Hugging Face Client --------------------
const hfClient = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_API_KEY,
});

// -------------------- Google Form Config --------------------
const GOOGLE_FORM_URL = "https://docs.google.com/forms/u/0/d/e/1FAIpQLSffbnGJGBC8awI_OgJF2HpLSvPOt6QgRrnBmPm6CwISGnAsoQ/formResponse";

const GOOGLE_FORM_ENTRIES = {
  question: "entry.2072247045",
  answer: "entry.455345515",
  timestamp: "entry.1378060286",
};

// -------------------- Google Form Logging Helper --------------------
async function logToGoogleForm(question, answer) {
  try {
    const formData = new URLSearchParams();
    formData.append(GOOGLE_FORM_ENTRIES.question, question);
    formData.append(GOOGLE_FORM_ENTRIES.answer, answer);
    formData.append(GOOGLE_FORM_ENTRIES.timestamp, new Date().toISOString());
    await axios.post(GOOGLE_FORM_URL, formData);
  } catch (err) {
    console.error("⚠️ Failed to log to Google Form:", err.message);
  }
}

// -------------------- Load knowledge base --------------------
let knowledge = [];
const knowledgePath = path.join(__dirname, "../knowledge.json");

async function loadKnowledge() {
  if (!knowledge.length) {
    const data = await fs.promises.readFile(knowledgePath, "utf-8");
    knowledge = JSON.parse(data);
  }
  return knowledge;
}

// -------------------- Helpers --------------------
function saveChat(userMsg, botReply) {
  const log = { timestamp: new Date().toISOString(), user: userMsg, bot: botReply };
  fs.appendFile("chatlogs.json", JSON.stringify(log) + "\n", (err) => {
    if (err) console.error("❌ Error saving chat:", err);
  });
}

function extractKeywords(text) {
  const tokenizer = new natural.WordTokenizer();
  return tokenizer.tokenize(text.toLowerCase());
}

// -------------------- AI-generated related terms --------------------
async function generateRelatedTerms(contextText, maxTerms = 15) {
  try {
    const prompt = `
You are a strict utility that returns related words/short phrases useful for keyword matching.
Given the TEXT between triple backticks, produce a JSON array (only the array, nothing else) of up to ${maxTerms} single-word or short-phrase related terms, synonyms, and closely related concepts that would help match this text to knowledge-base questions.
Respond ONLY with a JSON array of strings. Do NOT add explanation, commentary, or any extra text.

TEXT:
\`\`\`
${contextText}
\`\`\`
`;
    const resp = await hfClient.chat.completions.create({
      model: "google/gemma-2-2b-it:nebius",
      messages: [{ role: "user", content: prompt }],
      max_tokens: 200,
    });

    const raw = resp.choices?.[0]?.message?.content?.trim() || "";
    let parsed = null;
    try {
      parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) parsed = null;
    } catch {
      const match = raw.match(/\[.*\]/s);
      if (match) {
        try {
          parsed = JSON.parse(match[0]);
        } catch {
          parsed = null;
        }
      }
    }

    if (!parsed) {
      const lines = raw
        .split(/\r?\n/)
        .map((l) => l.trim().replace(/^[\-\*\d\.\)\s]+/, ""))
        .filter(Boolean)
        .slice(0, maxTerms);
      parsed = lines;
    }

    return Array.from(new Set(parsed.map((p) => String(p).toLowerCase().trim()))).slice(0, maxTerms);
  } catch (err) {
    console.error("⚠️ generateRelatedTerms failed:", err.message || err);
    return [];
  }
}

// -------------------- KB Matching --------------------
function findRelevantKnowledge(question, expandedKeywords = null) {
  const baseKeywords = extractKeywords(question);
  const keywordsSet = new Set(baseKeywords);

  if (expandedKeywords && Array.isArray(expandedKeywords)) {
    expandedKeywords.forEach((k) => {
      if (k && typeof k === "string") keywordsSet.add(k.toLowerCase());
    });
  }

  let bestMatch = null;
  let bestScore = 0;

  for (const entry of knowledge) {
    if (!entry.patterns) continue;
    for (const pattern of entry.patterns) {
      for (const keyword of keywordsSet) {
        const similarity = natural.JaroWinklerDistance(keyword, pattern.toLowerCase());
        if (similarity > bestScore) {
          bestScore = similarity;
          bestMatch = entry;
        }
      }
    }
  }

  return bestScore > 0.8 ? { entry: bestMatch, score: bestScore } : null;
}

// -------------------- Language Detection --------------------
async function detectLanguage(text) {
  try {
    const detection = await translate(text, { to: "en" }); // auto-detects source
    const detectedLang = detection.from?.language?.iso || "de";
    console.log("🌐 Detected language:", detectedLang);
    return detectedLang;
  } catch (err) {
    console.error("⚠️ Language detection failed:", err.message || err);
    return "de";
  }
}

// -------------------- Segment Protection --------------------
const FIXED_PROTECTED = [
  "Spielhof 9, 6317 Oberwil bei Zug, Switzerland",
  "info@labelmonster.swiss",
  "support@labelmonster.eu",
];

function protectSegments(text) {
  const mapping = {};
  let out = text;
  let idx = 0;

  for (const seg of FIXED_PROTECTED) {
    const re = new RegExp(seg.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
    out = out.replace(re, () => {
      const token = `__PROT_${idx}__`;
      mapping[token] = seg;
      idx += 1;
      return token;
    });
  }

  const emailRegex = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g;
  out = out.replace(emailRegex, (m) => {
    if (Object.values(mapping).includes(m)) return m;
    const token = `__PROT_${idx}__`;
    mapping[token] = m;
    idx += 1;
    return token;
  });

  const phoneRegex = /(\+?\d[\d ()-]{6,}\d)/g;
  out = out.replace(phoneRegex, (m) => {
    if (Object.values(mapping).includes(m)) return m;
    const token = `__PROT_${idx}__`;
    mapping[token] = m;
    idx += 1;
    return token;
  });

  return { protectedText: out, mapping };
}

function restoreSegments(text, mapping) {
  let out = text;
  for (const token in mapping) {
    const seg = mapping[token];
    const re = new RegExp(token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g");
    out = out.replace(re, seg);
  }
  return out;
}

// -------------------- Translation --------------------
async function translateText(text, targetLangCode) {
  if (!text) return text;
  try {
    const { protectedText, mapping } = protectSegments(text);
    const res = await translate(protectedText, { to: targetLangCode });
    const translated = typeof res === "string" ? res : res.text || protectedText;
    const restored = restoreSegments(translated, mapping);
    return restored;
  } catch (err) {
    console.error("❌ Translation error:", err.message || err);
    return text;
  }
}

// -------------------- Clean Model Output --------------------
function cleanModelOutput(raw) {
  if (!raw) return "";
  let out = raw;
  out = out.replace(/^["']+|["']+$/g, "").trim();
  out = out.replace(/\*\*/g, "").replace(/(^|\n)[ \t]*[-*+] /g, "$1");
  out = out.replace(/\r/g, "").replace(/\n{3,}/g, "\n\n").trim();
  return out;
}

// -------------------- Chat Endpoint --------------------
app.post("/api/chat", async (req, res) => {
  const { message } = req.body;
  if (!message) return res.json({ reply: "Keine Nachricht erhalten." });

  try {
    const userLang = await detectLanguage(message);

    const questionInGerman = userLang === "de" ? message : await translateText(message, "de");

    await loadKnowledge();

    const relatedTerms = await generateRelatedTerms(questionInGerman);
    const matchResult = findRelevantKnowledge(questionInGerman, relatedTerms);

    if (!matchResult) {
      const fallbackGerman =
        "Entschuldigung, das habe ich nicht verstanden. Bitte stellen Sie eine klare Frage oder senden Sie uns eine E-Mail an <no-translate>info@labelmonster.swiss</no-translate>, damit wir Ihnen besser weiterhelfen können.";
      const fallback = await translateText(fallbackGerman, userLang);
      saveChat(message, fallback);
      await logToGoogleForm(message, fallback);
      return res.json({ reply: fallback });
    }

    const matchedEntry = matchResult.entry;
    let kbAnswerGerman = cleanModelOutput(matchedEntry.answer || "");
    if (!kbAnswerGerman) {
      const fallbackGerman =
        "Entschuldigung, das habe ich nicht verstanden. Bitte stellen Sie eine klare Frage oder senden Sie uns eine E-Mail an <no-translate>info@labelmonster.swiss</no-translate>, damit wir Ihnen besser weiterhelfen können.";
      const fallback = await translateText(fallbackGerman, userLang);
      saveChat(message, fallback);
      await logToGoogleForm(message, fallback);
      return res.json({ reply: fallback });
    }

    let finalReply = kbAnswerGerman;
    if (userLang !== "de") {
      try {
        finalReply = await translateText(kbAnswerGerman, userLang);
      } catch (err) {
        console.error("⚠️ Translation of KB answer failed:", err.message || err);
        finalReply = kbAnswerGerman;
      }
    }

    finalReply = finalReply
      .replace(/__PROT_ADDR__/g, "Spielhof 9, 6317 Oberwil bei Zug, Switzerland")
      .replace(/__PROT_EMAIL__/g, "info@labelmonster.swiss")
      .replace(/\s{2,}/g, " ")
      .trim();

    saveChat(message, finalReply);
    await logToGoogleForm(message, finalReply);
    return res.json({ reply: finalReply });
  } catch (err) {
    console.error("❌ Chat error:", err.message || err);
    return res.json({ reply: "Fehler beim Abrufen der Antwort von Gemma." });
  }
});
