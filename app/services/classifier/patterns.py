"""
Trilingual regex pattern banks for intent classification.

Seven intent categories: conceptual_question, platform_query, legal_query,
document_query, bug_query, metadata_query, user_query.
"""

import re


# --- Legal queries ---
LEGAL_PATTERNS = [
    re.compile(
        r"\b(?:legal|laws?|regulations?|decrees?|provisions?|articles?|copyright|licens?e|privacy|gdpr|compliance)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:juridiques?|lois?|règlements?|décrets?|dispositions?|articles?|droit d'auteur|licence|"
        r"confidentialité|conformité)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:ال)?(?:"
        r"قانون|قوانين|تشريع|تشريعات|مرسوم|مراسيم|مادة|مواد|بند|بنود|تنظيم|حقوق|"
        r"ترخيص|مرخ(?:ص|صة)|خصوصية|لائحة|لوائح|بيانات|حماية|شروط|إجراءات|"
        r"نظام|أحكام|عقوبة|عقوبات|مخالف(?:ة|ات)|جزاء|جزاءات|مصنف(?:ات)?|"
        r"مؤلف|ملكية|فكرية"
        r")\w*\b"
    ),
    re.compile(r"\bis (?:it|this) legal\b", re.I),
    re.compile(r"\best[- ]ce (?:légal|autorisé)\b", re.I),
    re.compile(r"\bهل.*(?:قانوني|مشروع)\b"),
]

# --- Document queries (user-uploaded files) ---
# ONLY explicit document references trigger this intent.
# Generic verbs (explain, summarize) without document words do NOT match.
DOCUMENT_PATTERNS = [
    # English — explicit document references
    re.compile(r"\b(?:my (?:file|document|upload|pdf))\b", re.I),
    re.compile(r"\b(?:uploaded|summarize|summarise)\b.*\b(?:file|document|pdf)\b", re.I),
    re.compile(r"\bin (?:this|the|my) (?:document|file|pdf|paper|upload)\b", re.I),
    re.compile(r"\b(?:what does|according to) (?:this|the|my) (?:document|file|pdf|paper)\b", re.I),
    re.compile(r"\b(?:from|in) (?:the )?uploaded (?:file|document|pdf)\b", re.I),
    re.compile(r"\bsummarize my (?:pdf|document|file|paper)\b", re.I),
    re.compile(r"\bexplain (?:this|the) (?:document|paper|pdf|file)\b", re.I),
    # French — explicit document references
    re.compile(r"\b(?:mon (?:fichier|document|pdf))\b", re.I),
    re.compile(r"\b(?:résumer|analyser) (?:mon|le|ce) (?:fichier|document|pdf)\b", re.I),
    re.compile(r"\bdans (?:ce|le|mon) (?:document|fichier|pdf)\b", re.I),
    re.compile(r"\bque dit (?:ce|le|mon) (?:document|fichier)\b", re.I),
    re.compile(r"\bselon (?:ce|le|mon) (?:document|fichier|pdf)\b", re.I),
    # Arabic — explicit document references
    re.compile(r"\b(?:ملفي|مستندي|وثيقتي)\b"),
    re.compile(r"\bلخص\b.*\b(?:ملف|مستند|وثيقة)\b"),
    re.compile(r"\b(?:في|من|حسب) (?:هذا |هذه )?(?:الملف|المستند|الوثيقة|الـ ?pdf)\b"),
    re.compile(r"\bماذا (?:يقول|يذكر|يحتوي) (?:هذا )?(?:الملف|المستند)\b"),
]


# --- General knowledge / advice queries (should go directly to LLM) ---
GENERAL_KNOWLEDGE_PATTERNS = [
    # Greetings — short social messages that need no context/profile injection
    re.compile(
        r"^\s*(?:hi|hello|hey|yo|hiya|howdy|greetings|good\s*(?:morning|afternoon|evening|day)|thanks?(?:\s*you)?|thank\s*u|welcome)\s*[!?.]*\s*$",
        re.I,
    ),
    re.compile(
        r"^\s*(?:bonjour|salut|bonsoir|coucou|bonne\s*(?:journée|soirée)|merci)\s*[!?.]*\s*$",
        re.I,
    ),
    re.compile(
        r"^\s*(?:مرحبا|مرحبًا|سلام|أهلا|السلام عليكم|صباح الخير|مساء الخير|أهلاً|هلا|شكرا|شكراً)\s*[!?.]*\s*$"
    ),
    # Chatbot self-identity ("who are you", "what are you", etc.)
    re.compile(r"\bwho are you\b", re.I),
    re.compile(r"\bwhat are you\b", re.I),
    re.compile(r"\btell me about yourself\b", re.I),
    re.compile(r"\bintroduce yourself\b", re.I),
    re.compile(r"\bqui es[- ]tu\b", re.I),
    re.compile(r"\bqu'es[- ]tu\b", re.I),
    re.compile(r"\bprésentez?[- ](?:toi|vous)\b", re.I),
    re.compile(r"(?:من أنت|ما أنت|عرّف (?:عن )?نفسك)"),
    # English
    re.compile(
        r"\b(?:suggest|recommend|give me|create|make|build|design|write)\b"
        r".*\b(?:plan|roadmap|path|guide|schedule|curriculum|syllabus|strategy|steps)",
        re.I,
    ),
    re.compile(
        r"\b(?:how (?:to|do i|can i|should i))\b"
        r".*\b(?:learn|study|start|begin|master|improve|get into|get started)",
        re.I,
    ),
    re.compile(
        r"\b(?:tips?|advice|best (?:way|practice|approach)|tutorial)\b"
        r".*\b(?:learn|study|start|begin|master|improve)",
        re.I,
    ),
    re.compile(
        r"\b(?:learning|study) (?:plan|path|roadmap|guide|strategy)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:what (?:should i|do i need to)|where (?:should i|do i)) (?:learn|study|start|begin)\b",
        re.I,
    ),
    # Conversational / advisory — no retrieval needed
    re.compile(
        r"\b(?:how (?:to|do i|can i|should i))\b"
        r".*\b(?:build|create|make|design|develop|implement|write|set up|deploy)",
        re.I,
    ),
    re.compile(
        r"\b(?:brainstorm|ideate|ideas? for|think of|come up with)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:what is|what are|what's|explain|define|describe)\b"
        r".*\b(?:the (?:difference|concept|idea|purpose|role|meaning|definition))\b",
        re.I,
    ),
    re.compile(
        r"\b(?:compare|pros? and cons?|advantages?|disadvantages?|trade-?offs?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:can you|could you|help me)\b.*\b(?:explain|understand|clarify|elaborate)\b",
        re.I,
    ),
    # French
    re.compile(
        r"\b(?:suggérer|recommander|proposer|créer|faire|donner)\b"
        r".*\b(?:plan|parcours|programme|stratégie|étapes|guide)",
        re.I,
    ),
    re.compile(
        r"\b(?:comment)\b.*\b(?:apprendre|étudier|commencer|maîtriser|améliorer)",
        re.I,
    ),
    re.compile(
        r"\b(?:conseils?|astuces?|meilleure? (?:façon|méthode|approche))\b"
        r".*\b(?:apprendre|étudier|commencer)",
        re.I,
    ),
    re.compile(
        r"\b(?:plan|parcours|programme) (?:d'apprentissage|d'étude|de formation)\b",
        re.I,
    ),
    # Arabic
    re.compile(
        r"\b(?:اقترح|أنشئ|صمم|ضع|اكتب|أعطني)\b"
        r".*\b(?:خطة|مسار|برنامج|استراتيجية|خطوات|دليل)",
    ),
    re.compile(
        r"\b(?:كيف)\b.*\b(?:أتعلم|أدرس|أبدأ|أتقن|أحسن)",
    ),
    re.compile(
        r"\b(?:نصائح|أفضل (?:طريقة|أسلوب|نهج))\b"
        r".*\b(?:تعلم|دراسة|بدء)",
    ),
    re.compile(
        r"\b(?:خطة|مسار|برنامج) (?:تعلم|دراسة|تدريب)\b",
    ),
    # Arabic conversational / advisory
    re.compile(r"\b(?:كيف (?:أبني|أصمم|أنشئ|أطور|أكتب))\b"),
    re.compile(r"\b(?:أفكار|اقتراحات|عصف ذهني)\b"),
    # French conversational / advisory
    re.compile(
        r"\b(?:comment)\b.*\b(?:construire|créer|développer|concevoir|implémenter|déployer)",
        re.I,
    ),
    re.compile(
        r"\b(?:idées?|brainstorm|réfléchir)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:comparer|avantages?|inconvénients?|différences?)\b",
        re.I,
    ),
    # Advisory / rule-based / theoretical — prevents platform keyword leak
    # English
    re.compile(
        r"\b(?:what|which)\b.*\b(?:rules?|principles?|guidelines?|best practices?|standards?|criteria|ethics?|norms?)\b",
        re.I,
    ),
    re.compile(r"\b(?:how|why)\s+should\b", re.I),
    # French
    re.compile(
        r"\b(?:quelles?|quels?)\b.*\b(?:règles?|principes?|lignes directrices|bonnes pratiques|normes?|critères?|éthiques?)\b",
        re.I,
    ),
    re.compile(r"\b(?:comment|pourquoi)\s+(?:devrait|faut[- ]il|doit[- ]on)\b", re.I),
    # Arabic
    re.compile(r"\b(?:ما هي|ما)\b.*\b(?:قواعد|مبادئ|إرشادات|معايير|أخلاقيات|ممارسات)\b"),
    re.compile(r"\b(?:كيف|لماذا)\s+(?:يجب|ينبغي)\b"),
]


# Soft document hints — DISABLED.
# Was causing false positives: generic verbs like "explain" or "summarize"
# triggered document_query even for conceptual questions.
# SOFT_DOCUMENT_PATTERN = re.compile(
#     r"\b(?:summarize|summarise|explain|analyze|analyse|extract|"
#     r"résumer|expliquer|analyser|لخص|اشرح)\b",
#     re.I,
# )


