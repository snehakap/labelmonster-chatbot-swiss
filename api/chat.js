
// ======================= FULL SERVER CODE =======================

import path from "path";
import { promises as fs } from "fs";
import { OpenAI } from "openai";
import axios from "axios";
import translate from "translate-google";

// Words to ignore
  global.stopwords = [
  // Articles & determiners
  "der","die","das","den","dem","des","ein","eine","einen","einem","einer",'etwas',
  "dies","diese","dieser","dieses","jenes","jene","jener","solche","erzähl","geht",
  "manche","alle","jede","jeder","jedes","es","gibt","aus","brauche","meine","stellen","erzählen",
  // Pronouns
  "ich","du","er","sie","es","wir","ihr","man",
  "mich","dich","ihn","sie","uns","euch","ihnen","ihm","ihr","für","mein","mehr",

  // Question words
  "wie","wo","was","wer","wen","wem","wessen","möglich","bieten", "Sie" ,"an",
  "welche","welcher","welches","warum","wieso","weshalb","wohin","woher",

  // Prepositions
  "in","im","ins","am","an","auf","für","von","mit","ohne","über","unter",
  "bei","durch","gegen","um","zu","zum","zur","nach","vor","hinter",
  "neben","zwischen","entlang","außer","innerhalb","außerhalb",

  // Auxiliary verbs
  "bin","bist","ist","sind","seid","war","waren","wirst","wurde","wurden",
  "habe","hast","hat","haben","habt","hatte","hatten",

  // Modal verbs
  "kann","kannst","können","könnt","könnte",
  "muss","musst","müssen","müsst","müsste",
  "soll","sollst","sollen","sollt","sollte","sollten",
  "darf","darfst","dürfen","dürft","dürfte",
  "will","willst","wollen","wollt","wollte","wollten",
  "möchte","möchtest","möchten","möchtet",

  // Common verbs that never represent the subject
  "finden","prüfen","sehen","anzeigen","zeigen","bekommen","holen",
  "machen","brauchen","suchen","geben","nehmen","gehen","kommen",
  "erhalten","laden","vergleichen",
  "nutzen","benutzen","verwenden","welchen",

  // Generic chatbot words (never subjects)
  "bitte","danke",

  // Adverbs / filler words
  "so","auch","nur","schon","noch","dann","danach","jetzt","heute",
  "gestern","morgen","bald","gleich","hier","dort","da","mal","nun","basierend", "basiert", "empfehlen", "budget",

  // Conjunctions
  "und","oder","aber","doch","jedoch","denn","falls","wenn","weil",
  "ob","beziehungsweise","bzw",

  // Other non-subject particles
  "ja","nein","okay","ok","mal","eben","halt","gern","mir","ihre"
];

function protectCPMWords(text) {
  const safeWords = [
    "CPM-200","CPM200","CPM 200","CPM-100","CPM100","CPM 100",
    "cpm-200","cpm200","cpm 200","cpm-100","cpm100","cpm 100"
  ];

  safeWords.forEach((w, i) => {
    const token = `__SAFE_CPM_${i}__`;
    text = text.replace(new RegExp(w, "gi"), token);
  });

  return { text, safeWords };
}

function restoreCPMWords(text, safeWords) {
  safeWords.forEach((w, i) => {
    const token = `__SAFE_CPM_${i}__`;
    text = text.replace(new RegExp(token, "g"), w);
  });
  return text;
}

// -------------------- Load knowledge.json --------------------
let knowledge = [];
const knowledgePath = path.join(process.cwd(), "knowledge.json");

async function loadKnowledge() {
  if (!knowledge.length) {
    const data = await fs.readFile(knowledgePath, "utf-8");
    knowledge = JSON.parse(data);
  }
  return knowledge;
}


// -------------------- Utility Functions --------------------
function extractKeywords(text) {

    // 1. Lowercase + keep umlauts + remove punctuation
    const rawWords = text
        .toLowerCase()
        .replace(/[^a-zA-ZäöüÄÖÜß]/g, " ")
        .split(/\s+/)
        .filter(Boolean);


    // 2. Remove fragments shorter than 3 letters
    let minLenWords = rawWords.filter(w => w.length >= 3);

    // 3. Remove greetings inside long sentences
    const greetingWords = ["hallo", "hi", "hello", "guten", "tag", "morgen"];
    if (minLenWords.length > 1 && greetingWords.includes(minLenWords[0])) {
        minLenWords.shift();
    }

    // 4. Apply global stopwords
    let keywords = minLenWords.filter(w => !global.stopwords.includes(w));

    // 5. Ignore certain words ONLY IF keyword count > 1
    const ignoreIfMultiple = [
        "etiketten", "drucker", "gerät", "geräte",
        "artikel", "etikett", "drucksysteme",
        "cpm", "druckt", "software", "informationen"
    ];

    if (keywords.length > 1) {
        keywords = keywords.filter(w => !ignoreIfMultiple.includes(w));
    }

    return keywords;
}

async function findRelevantKnowledge(question) {
  const knowledgeData = await loadKnowledge();

  const keywords = extractKeywords(question)
  .map(k => k.toLowerCase())
  .filter(k => !global.stopwords.includes(k));
  if (keywords.length === 0) {
    return { bestMatch: null, bestScore: 0 };
  }

  let bestMatch = null;
  let bestScore = 0;

  for (const entry of knowledgeData) {
    if (!entry.patterns) continue;

    let entryScore = 0;   // total keyword matches across all patterns in this ID

    for (const pattern of entry.patterns) {
      const p = pattern.toLowerCase();

      keywords.forEach(kw => {
        const occurrences = p.split(kw).length - 1; // count matches
        if (occurrences > 0) entryScore += occurrences;
      });
    }

    // Choose the ID with the HIGHEST total score
    if (entryScore > bestScore) {
      bestScore = entryScore;
      bestMatch = entry;
    }
  }

  if (bestScore === 0) return { bestMatch: null, bestScore: 0 };

  return { bestMatch, bestScore };
}

function extractSentenceSubject(sentence) {
    // Normalize
    let cleaned = sentence
        .toLowerCase()
        .replace(/[^a-zA-Z0-9äöüÄÖÜß]/g, " ")
        .trim();

    const words = cleaned.split(/\s+/).filter(Boolean);

    // 1️⃣ Remove stopwords
    let meaningful = words.filter(w => !global.stopwords.includes(w));

    // 2️⃣ Remove very short fragments (<3 or <4 as needed)
    meaningful = meaningful.filter(w => w.length >= 3);

    // 3️⃣ NEW — Remove generic industry terms ONLY IF more than 1 keyword
    const ignoreIfMultiple = [
        "etiketten", "drucker", "gerät", "geräte",
        "artikel", "etikett", "drucksysteme",
        "cpm", "druckt", "software"
    ];

    if (meaningful.length > 1) {
        meaningful = meaningful.filter(w => !ignoreIfMultiple.includes(w));
    }

    // 4️⃣ Return first meaningful → This is your subject
    const subject = meaningful[0] || "";
    return subject;
}

// -------------------- Hugging Face Client --------------------
const hfClient = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_API_KEY,
});

// -------------------- Google Form Config --------------------
const GOOGLE_FORM_URL =
  "https://docs.google.com/forms/u/0/d/e/1FAIpQLSffbnGJGBC8awI_OgJF2HpLSvPOt6QgRrnBmPm6CwISGnAsoQ/formResponse";

const GOOGLE_FORM_ENTRIES = {
  question: "entry.2072247045",
  answer: "entry.455345515",
  timestamp: "entry.1378060286",
};

// -------------------- Segment Protection Helpers --------------------
function protectSegments(text) {
  // Match:
  // 1. HTML tags
  // 2. URLs
  // 3. Emails
  // 4. Physical addresses
  const regex = /<[^>]+>|https?:\/\/\S+|\b[\w.-]+@[\w.-]+\.\w{2,}\b|\b[A-ZÄÖÜ][a-zäöüß]+\s\d{1,3},\s\d{4,5}\s[A-ZÄÖÜa-zäöüß\s]+,?\s?[A-ZÄÖÜa-zäöüß]*\b/g;

  const mapping = {};
  let i = 0;

  const protectedText = text.replace(regex, (match) => {
    const key = `__SEG_${i}__`;
    mapping[key] = match;
    i++;
    return key;
  });

  return { protectedText, mapping };
}

function restoreSegments(text, mapping) {
  let restored = text;
  for (const key in mapping) {
    restored = restored.replace(key, mapping[key]);
  }
  return restored;
}

// -------------------- TRANSLATION --------------------
async function translateText(text, targetLangCode) {
  if (!text) return text;

  try {
    // Words that must NOT be translated
    const safeWords = [
      "CPM-200","CPM200","CPM 200","CPM-100","CPM100","CPM 100",
      "cpm-200","cpm200","cpm 200","cpm-100","cpm100","cpm 100"
    ];

    // Protect CPM words
    safeWords.forEach((w, i) => {
      const token = `__SAFE_CPM_${i}__`;
      text = text.replace(new RegExp(w, "g"), token);
    });

    // Protect HTML links & tags
    const { protectedText, mapping } = protectSegments(text);

    // Translate
    let translated = await translate(protectedText, { to: targetLangCode });
    translated = typeof translated === "string" ? translated : translated.text;

    // Restore HTML segments
    translated = restoreSegments(translated, mapping);

    // Restore CPM tokens
    safeWords.forEach((w, i) => {
      const token = `__SAFE_CPM_${i}__`;
      translated = translated.replace(new RegExp(token, "g"), w);
    });
    
    return translated;

  } catch (err) {
    console.error("❌ Translation error:", err.message);
    return text;
  }
}

// -------------------- Log to Google Form --------------------
async function logToGoogleForm(question, answer) {
  try {
    const payload = new URLSearchParams();
    payload.append(GOOGLE_FORM_ENTRIES.question, question);
    payload.append(GOOGLE_FORM_ENTRIES.answer, answer);

    await axios.post(GOOGLE_FORM_URL, payload.toString(), {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });

  } catch (err) {
    console.error("⚠️ Error sending to Google Form:", err.message);
  }
}

// -------------------- HELPER FOR VERCEL RESPONSE --------------------
function sendJSON(res, status, obj) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(obj));
}

// -------------------- EXPORT DEFAULT (REQUIRED BY VERCEL) --------------------
export default async function handler(req, res) {

  // -------------------- CORS --------------------
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return sendJSON(res, 200, {});
  }

  // -------------------- ONLY POST REACHES HERE --------------------
  if (req.method !== "POST") {
    return sendJSON(res, 405, { error: "Method not allowed" });
  }

  // Parse POST body (Vercel gives buffer)
  const body = req.body || JSON.parse(await new Promise(resolve => {
    let data = "";
    req.on("data", chunk => (data += chunk));
    req.on("end", () => resolve(data));
  }));

  const message = body.message || "";
  const lang = body.language || "de";

  // -------------------- CPM Protection --------------------
  const { text: protectedMsg, safeWords } = protectCPMWords(message);

  const germanMessage = await translateText(protectedMsg, "de");

  // -------------------- EXACT MATCH LOGIC (unchanged) --------------------

  await loadKnowledge();

  let exactKBMatch = null;
  const cleanedGerman = germanMessage.trim().toLowerCase();

  for (const entry of knowledge) {
    if (!entry.patterns) continue;

    for (const pattern of entry.patterns) {
      const cleanedPattern = pattern.trim().toLowerCase();

      if (cleanedGerman === cleanedPattern) {
        exactKBMatch = entry;
        break;
      }

      const gWords = cleanedGerman.split(" ").length;
      const pWords = cleanedPattern.split(" ").length;

      if (gWords === pWords && cleanedGerman.includes(cleanedPattern)) {
        exactKBMatch = entry;
        break;
      }
    }

    if (exactKBMatch) break;
  }

  if (exactKBMatch) {

    let replyGerman = exactKBMatch.answer;

    const { text: protectedReply, safeWords: replySafeWords } = protectCPMWords(replyGerman);
    let finalReply =
      lang === "de" ? replyGerman : await translateText(protectedReply, lang);

    finalReply = restoreCPMWords(finalReply, replySafeWords);

    await logToGoogleForm(message, finalReply);
    return sendJSON(res, 200, { reply: finalReply });
  }

  // -------------------- SUBJECT MATCH (unchanged) --------------------

  const extractedSubject = extractSentenceSubject(germanMessage);

  if (extractedSubject) {
    const kbData = await loadKnowledge();

    const subjectMatch = kbData.find(entry =>
      entry.subject &&
      entry.subject.some(s => s.toLowerCase() === extractedSubject.toLowerCase())
    );

    if (subjectMatch) {
      const replyGerman = subjectMatch.answer;
      let finalReply = lang === "de" ? replyGerman : await translateText(replyGerman, lang);
      finalReply = restoreCPMWords(finalReply, safeWords);
      await logToGoogleForm(message, finalReply);
      return sendJSON(res, 200, { reply: finalReply });
    }
  }

  // -------------------- SIMILARITY 80% (unchanged) --------------------

  let { bestMatch, bestScore } = await findRelevantKnowledge(germanMessage);

  if (bestMatch && bestScore >= 0.8) {
    const replyGerman = bestMatch.answer;
    let finalReply = lang === "de" ? replyGerman : await translateText(replyGerman, lang);
    finalReply = restoreCPMWords(finalReply, safeWords);
    await logToGoogleForm(message, finalReply);
    return sendJSON(res, 200, { reply: finalReply });
  }


  // -------------------- FALLBACK (unchanged) --------------------
  const fallbackGerman =
    "Entschuldigung, das habe ich nicht verstanden. Bitte stellen Sie eine klare Frage oder senden Sie uns eine E-Mail an <a href='mailto:info@labelmonster.swiss'>info@labelmonster.swiss</a>.";

  const translatedFallback =
    lang === "de" ? fallbackGerman : await translateText(fallbackGerman, lang);

  const finalReply = restoreCPMWords(translatedFallback, safeWords);

  await logToGoogleForm(message, finalReply);
  return sendJSON(res, 200, { reply: finalReply });
}

