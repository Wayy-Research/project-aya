"""Test fixtures and sample data for multilingual testing.

Covers all 70+ languages from CohereLabs/tiny-aya-global.
Includes reasoning-specific prompts across four categories:
mathematical, causal, analogical, and logical.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# General knowledge prompts across all 70+ target languages.
# Each key is a prompt category; each value maps lang_code -> prompt.
# ---------------------------------------------------------------------------
MULTILINGUAL_PROMPTS: dict[str, dict[str, str]] = {
    "general_knowledge": {
        # ---- Romance ----
        "en": "Explain the concept of gravity in simple terms.",
        "nl": "Leg het concept van zwaartekracht uit in eenvoudige woorden.",
        "fr": "Expliquez le concept de la gravité en termes simples.",
        "it": "Spiega il concetto di gravità in termini semplici.",
        "pt": "Explique o conceito de gravidade em termos simples.",
        "ro": "Explicați conceptul de gravitație în termeni simpli.",
        "es": "Explica el concepto de gravedad en términos sencillos.",
        "ca": "Explica el concepte de gravetat en termes senzills.",
        "gl": "Explica o concepto de gravidade en termos sinxelos.",
        # ---- Germanic ----
        "de": "Erkläre das Konzept der Schwerkraft in einfachen Worten.",
        "da": "Forklar begrebet tyngdekraft med enkle ord.",
        "sv": "Förklara begreppet gravitation med enkla ord.",
        "no": "Forklar begrepet tyngdekraft med enkle ord.",
        # ---- Slavic ----
        "cs": "Vysvětlete koncept gravitace jednoduchými slovy.",
        "pl": "Wyjaśnij pojęcie grawitacji prostymi słowami.",
        "uk": "Поясніть концепцію гравітації простими словами.",
        "ru": "Объясните концепцию гравитации простыми словами.",
        "hr": "Objasnite koncept gravitacije jednostavnim riječima.",
        "sk": "Vysvetlite koncept gravitácie jednoduchými slovami.",
        "sl": "Razložite koncept gravitacije s preprostimi besedami.",
        "sr": "Објасните концепт гравитације једноставним речима.",
        "bg": "Обяснете концепцията за гравитацията с прости думи.",
        # ---- Baltic ----
        "lv": "Izskaidrojiet gravitācijas jēdzienu vienkāršiem vārdiem.",
        "lt": "Paaiškinkite gravitacijos sąvoką paprastais žodžiais.",
        # ---- Hellenic / Uralic / Finnic / Hungarian ----
        "el": "Εξηγήστε την έννοια της βαρύτητας με απλά λόγια.",
        "et": "Selgitage gravitatsiooni mõistet lihtsate sõnadega.",
        "fi": "Selitä painovoiman käsite yksinkertaisin sanoin.",
        "hu": "Magyarázza el a gravitáció fogalmát egyszerű szavakkal.",
        # ---- Basque / Celtic / Maltese ----
        "eu": "Azaldu grabitate kontzeptua hitz sinpleetan.",
        "cy": "Eglurwch y cysyniad o ddisgyrchiant mewn geiriau syml.",
        "ga": "Mínigh coincheap an imtharraingthe i dtéarmaí simplí.",
        "mt": "Spjega l-kunċett tal-gravità fi kliem sempliċi.",
        # ---- Semitic / Iranic / Turkic ----
        "ar": "اشرح مفهوم الجاذبية بعبارات بسيطة.",
        "fa": "مفهوم گرانش را به زبان ساده توضیح دهید.",
        "ur": "کشش ثقل کے تصور کو آسان الفاظ میں بیان کریں۔",
        "tr": "Yerçekimi kavramını basit terimlerle açıklayın.",
        "he": "הסבירו את מושג הכבידה במילים פשוטות.",
        # ---- Indo-Aryan ----
        "hi": "गुरुत्वाकर्षण की अवधारणा को सरल शब्दों में समझाइए।",
        "mr": "गुरुत्वाकर्षणाची संकल्पना सोप्या शब्दांत समजावून सांगा.",
        "bn": "মাধ্যাকর্ষণের ধারণাটি সহজ ভাষায় ব্যাখ্যা করুন।",
        "gu": "ગુરુત્વાકર્ષણની વિભાવના સરળ શબ્દોમાં સમજાવો.",
        "pa": "ਗੁਰੂਤਾ ਦੀ ਧਾਰਨਾ ਨੂੰ ਸਰਲ ਸ਼ਬਦਾਂ ਵਿੱਚ ਸਮਝਾਓ।",
        "ne": "गुरुत्वाकर्षणको अवधारणालाई सरल शब्दमा व्याख्या गर्नुहोस्।",
        # ---- Dravidian ----
        "ta": "புவியீர்ப்பு என்ற கருத்தை எளிய சொற்களில் விளக்குங்கள்.",
        "te": "గురుత్వాకర్షణ భావనను సరళమైన పదాలలో వివరించండి.",
        # ---- Austronesian ----
        "tl": "Ipaliwanag ang konsepto ng grabidad sa simpleng salita.",
        "ms": "Terangkan konsep graviti dalam istilah mudah.",
        "id": "Jelaskan konsep gravitasi dengan istilah sederhana.",
        "jv": "Jelasna konsep gravitasi nganggo tembung sing gampang.",
        # ---- Mainland Southeast Asian ----
        "vi": "Giải thích khái niệm trọng lực bằng thuật ngữ đơn giản.",
        "km": "ពន្យល់គោលគំនិតនៃទំនាញផែនដីដោយប្រើពាក្យសាមញ្ញ។",
        "th": "อธิบายแนวคิดเรื่องแรงโน้มถ่วงด้วยคำง่ายๆ",
        "lo": "ອະທິບາຍແນວຄິດຂອງແຮງດຶງດູດດ້ວຍຄຳສັບງ່າຍໆ.",
        "my": "ဆွဲငင်အားသဘောတရားကို ရိုးရှင်းသော စကားလုံးများဖြင့် ရှင်းပြပါ။",
        # ---- CJK ----
        "zh": "用简单的语言解释引力的概念。",
        "ja": "重力の概念を簡単な言葉で説明してください。",
        "ko": "중력의 개념을 간단한 용어로 설명해 주세요.",
        # ---- African ----
        "am": "የስበት ጽንሰ-ሐሳብን በቀላል ቃላት ያብራሩ።",
        "ha": "Bayyana manufar nauyi a cikin kalmomi masu sauƙi.",
        "ig": "Kọwaa echiche nke gravity n'okwu dị mfe.",
        "mg": "Hazavao amin'ny teny tsotra ny foto-kevitry ny gravité.",
        "sn": "Tsanangura pfungwa yegravity nemashoko asingaomeseri.",
        "sw": "Eleza dhana ya mvutano kwa maneno rahisi.",
        "wo": "Nëblalal mbir mi ci ay baat yu yomb.",
        "xh": "Cacisa umxholo wamandla amtsalane ngamazwi alula.",
        "yo": "Ṣe àlàyé èrò àdánidá ní àwọn ọ̀rọ̀ tó rọrùn.",
        "zu": "Chaza umqondo wesisindo ngamagama alula.",
    },
}

# ---------------------------------------------------------------------------
# Tool schemas for tool calling tests - matching common function calling
# formats.
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "translate",
        "description": "Translate text between languages",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source_lang": {"type": "string"},
                "target_lang": {"type": "string"},
            },
            "required": ["text", "target_lang"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool calling scenarios per language.
# Each scenario asks the model to call get_weather for a major city
# associated with that language's primary country/region.
# ---------------------------------------------------------------------------
TOOL_SCENARIOS: dict[str, dict] = {
    # ---- English ----
    "en": {
        "prompt": (
            "You have these tools: get_weather, search_web, translate. "
            'The user says: "What is the weather in London?" '
            "Respond with a JSON tool call."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "London"},
    },
    # ---- Dutch ----
    "nl": {
        "prompt": (
            "Je hebt deze tools: get_weather, search_web, translate. "
            'De gebruiker zegt: "Wat is het weer in Amsterdam?" '
            "Antwoord met een JSON tool call."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Amsterdam"},
    },
    # ---- French ----
    "fr": {
        "prompt": (
            "Vous disposez de ces outils : get_weather, search_web, translate. "
            "L'utilisateur dit : \"Quel temps fait-il à Paris ?\" "
            "Répondez avec un appel d'outil JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Paris"},
    },
    # ---- Italian ----
    "it": {
        "prompt": (
            "Hai questi strumenti: get_weather, search_web, translate. "
            'L\'utente dice: "Com\'è il meteo a Roma?" '
            "Rispondi con una chiamata strumento JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Rome"},
    },
    # ---- Portuguese ----
    "pt": {
        "prompt": (
            "Você tem estas ferramentas: get_weather, search_web, translate. "
            'O usuário diz: "Como está o tempo em Lisboa?" '
            "Responda com uma chamada de ferramenta JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Lisbon"},
    },
    # ---- Romanian ----
    "ro": {
        "prompt": (
            "Aveți aceste instrumente: get_weather, search_web, translate. "
            'Utilizatorul spune: "Cum este vremea în București?" '
            "Răspundeți cu un apel de instrument JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Bucharest"},
    },
    # ---- Spanish ----
    "es": {
        "prompt": (
            "Tienes estas herramientas: get_weather, search_web, translate. "
            'El usuario dice: "¿Cuál es el clima en Madrid?" '
            "Responde con una llamada JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Madrid"},
    },
    # ---- Catalan ----
    "ca": {
        "prompt": (
            "Tens aquestes eines: get_weather, search_web, translate. "
            "L'usuari diu: \"Quin temps fa a Barcelona?\" "
            "Respon amb una crida d'eina JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Barcelona"},
    },
    # ---- Galician ----
    "gl": {
        "prompt": (
            "Tes estas ferramentas: get_weather, search_web, translate. "
            'O usuario di: "Cal é o tempo en Santiago?" '
            "Responde cunha chamada de ferramenta JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Santiago"},
    },
    # ---- German ----
    "de": {
        "prompt": (
            "Du hast diese Werkzeuge: get_weather, search_web, translate. "
            'Der Benutzer sagt: "Wie ist das Wetter in Berlin?" '
            "Antworte mit einem JSON-Tool-Aufruf."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Berlin"},
    },
    # ---- Danish ----
    "da": {
        "prompt": (
            "Du har disse værktøjer: get_weather, search_web, translate. "
            'Brugeren siger: "Hvordan er vejret i København?" '
            "Svar med et JSON-værktøjskald."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Copenhagen"},
    },
    # ---- Swedish ----
    "sv": {
        "prompt": (
            "Du har dessa verktyg: get_weather, search_web, translate. "
            'Användaren säger: "Hur är vädret i Stockholm?" '
            "Svara med ett JSON-verktygsanrop."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Stockholm"},
    },
    # ---- Norwegian ----
    "no": {
        "prompt": (
            "Du har disse verktøyene: get_weather, search_web, translate. "
            'Brukeren sier: "Hvordan er været i Oslo?" '
            "Svar med et JSON-verktøykall."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Oslo"},
    },
    # ---- Czech ----
    "cs": {
        "prompt": (
            "Máte tyto nástroje: get_weather, search_web, translate. "
            'Uživatel říká: "Jaké je počasí v Praze?" '
            "Odpovězte voláním nástroje ve formátu JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Prague"},
    },
    # ---- Polish ----
    "pl": {
        "prompt": (
            "Masz te narzędzia: get_weather, search_web, translate. "
            'Użytkownik mówi: "Jaka jest pogoda w Warszawie?" '
            "Odpowiedz wywołaniem narzędzia JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Warsaw"},
    },
    # ---- Ukrainian ----
    "uk": {
        "prompt": (
            "У вас є інструменти: get_weather, search_web, translate. "
            'Користувач каже: "Яка погода в Києві?" '
            "Відповідайте викликом інструмента JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Kyiv"},
    },
    # ---- Russian ----
    "ru": {
        "prompt": (
            "У вас есть инструменты: get_weather, search_web, translate. "
            'Пользователь говорит: "Какая погода в Москве?" '
            "Ответьте вызовом инструмента JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Moscow"},
    },
    # ---- Croatian ----
    "hr": {
        "prompt": (
            "Imate ove alate: get_weather, search_web, translate. "
            'Korisnik kaže: "Kakvo je vrijeme u Zagrebu?" '
            "Odgovorite JSON pozivom alata."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Zagreb"},
    },
    # ---- Slovak ----
    "sk": {
        "prompt": (
            "Máte tieto nástroje: get_weather, search_web, translate. "
            'Používateľ hovorí: "Aké je počasie v Bratislave?" '
            "Odpovedzte volaním nástroja vo formáte JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Bratislava"},
    },
    # ---- Slovenian ----
    "sl": {
        "prompt": (
            "Imate ta orodja: get_weather, search_web, translate. "
            'Uporabnik pravi: "Kakšno je vreme v Ljubljani?" '
            "Odgovorite s klicem orodja JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Ljubljana"},
    },
    # ---- Serbian ----
    "sr": {
        "prompt": (
            "Имате ове алате: get_weather, search_web, translate. "
            'Корисник каже: "Какво је време у Београду?" '
            "Одговорите JSON позивом алата."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Belgrade"},
    },
    # ---- Bulgarian ----
    "bg": {
        "prompt": (
            "Имате тези инструменти: get_weather, search_web, translate. "
            'Потребителят казва: "Какво е времето в София?" '
            "Отговорете с JSON извикване на инструмент."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Sofia"},
    },
    # ---- Latvian ----
    "lv": {
        "prompt": (
            "Jums ir šie rīki: get_weather, search_web, translate. "
            'Lietotājs saka: "Kāds ir laiks Rīgā?" '
            "Atbildiet ar JSON rīka izsaukumu."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Riga"},
    },
    # ---- Lithuanian ----
    "lt": {
        "prompt": (
            "Turite šiuos įrankius: get_weather, search_web, translate. "
            'Vartotojas sako: "Koks oras Vilniuje?" '
            "Atsakykite JSON įrankio iškvietimu."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Vilnius"},
    },
    # ---- Greek ----
    "el": {
        "prompt": (
            "Έχετε αυτά τα εργαλεία: get_weather, search_web, translate. "
            'Ο χρήστης λέει: "Ποιος είναι ο καιρός στην Αθήνα;" '
            "Απαντήστε με κλήση εργαλείου JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Athens"},
    },
    # ---- Estonian ----
    "et": {
        "prompt": (
            "Teil on need tööriistad: get_weather, search_web, translate. "
            'Kasutaja ütleb: "Milline on ilm Tallinnas?" '
            "Vastake JSON tööriista kutsega."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Tallinn"},
    },
    # ---- Finnish ----
    "fi": {
        "prompt": (
            "Sinulla on nämä työkalut: get_weather, search_web, translate. "
            'Käyttäjä sanoo: "Millainen sää on Helsingissä?" '
            "Vastaa JSON-työkalukutsulla."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Helsinki"},
    },
    # ---- Hungarian ----
    "hu": {
        "prompt": (
            "Ezek az eszközei vannak: get_weather, search_web, translate. "
            'A felhasználó azt mondja: "Milyen az időjárás Budapesten?" '
            "Válaszoljon JSON eszközhívással."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Budapest"},
    },
    # ---- Basque ----
    "eu": {
        "prompt": (
            "Tresna hauek dituzu: get_weather, search_web, translate. "
            'Erabiltzaileak dio: "Zein da eguraldia Bilbon?" '
            "Erantzun JSON tresna deialdiarekin."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Bilbao"},
    },
    # ---- Welsh ----
    "cy": {
        "prompt": (
            "Mae gennych yr offer hyn: get_weather, search_web, translate. "
            "Y defnyddiwr yn dweud: \"Beth yw'r tywydd yng Nghaerdydd?\" "
            "Atebwch gyda galwad offeryn JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Cardiff"},
    },
    # ---- Irish ----
    "ga": {
        "prompt": (
            "Tá na huirlisí seo agat: get_weather, search_web, translate. "
            'Deir an t-úsáideoir: "Cén aimsir atá i mBaile Átha Cliath?" '
            "Freagair le glao uirlise JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Dublin"},
    },
    # ---- Maltese ----
    "mt": {
        "prompt": (
            "Għandek dawn l-għodod: get_weather, search_web, translate. "
            'L-utent jgħid: "Xi temp jagħmel il-Belt Valletta?" '
            "Wieġeb b'sejħa ta' għodda JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Valletta"},
    },
    # ---- Arabic ----
    "ar": {
        "prompt": (
            "لديك هذه الأدوات: get_weather, search_web, translate. "
            'المستخدم يقول: "ما الطقس في القاهرة؟" '
            "أجب بنداء أداة JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Cairo"},
    },
    # ---- Persian ----
    "fa": {
        "prompt": (
            "شما این ابزارها را دارید: get_weather, search_web, translate. "
            'کاربر می\u200cگوید: "آب و هوای تهران چطور است؟" '
            "با یک فراخوانی ابزار JSON پاسخ دهید."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Tehran"},
    },
    # ---- Urdu ----
    "ur": {
        "prompt": (
            "آپ کے پاس یہ ٹولز ہیں: get_weather, search_web, translate۔ "
            'صارف کہتا ہے: "اسلام آباد کا موسم کیسا ہے؟" '
            "JSON ٹول کال سے جواب دیں۔"
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Islamabad"},
    },
    # ---- Turkish ----
    "tr": {
        "prompt": (
            "Şu araçlarınız var: get_weather, search_web, translate. "
            "Kullanıcı diyor ki: \"İstanbul'da hava nasıl?\" "
            "JSON araç çağrısı ile yanıtlayın."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Istanbul"},
    },
    # ---- Hebrew ----
    "he": {
        "prompt": (
            "יש לך את הכלים הבאים: get_weather, search_web, translate. "
            'המשתמש אומר: "מה מזג האוויר בירושלים?" '
            "ענה עם קריאת כלי JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Jerusalem"},
    },
    # ---- Hindi ----
    "hi": {
        "prompt": (
            "आपके पास ये टूल हैं: get_weather, search_web, translate। "
            'उपयोगकर्ता कहता है: "दिल्ली का मौसम क्या है?" '
            "JSON टूल कॉल से जवाब दें।"
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Delhi"},
    },
    # ---- Marathi ----
    "mr": {
        "prompt": (
            "तुमच्याकडे ही साधने आहेत: get_weather, search_web, translate. "
            'वापरकर्ता म्हणतो: "मुंबईचे हवामान कसे आहे?" '
            "JSON साधन कॉलने उत्तर द्या."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Mumbai"},
    },
    # ---- Bengali ----
    "bn": {
        "prompt": (
            "আপনার কাছে এই টুলগুলো আছে: get_weather, search_web, translate। "
            'ব্যবহারকারী বলেন: "ঢাকার আবহাওয়া কেমন?" '
            "JSON টুল কল দিয়ে উত্তর দিন।"
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Dhaka"},
    },
    # ---- Gujarati ----
    "gu": {
        "prompt": (
            "તમારી પાસે આ સાધનો છે: get_weather, search_web, translate. "
            'વપરાશકર્તા કહે છે: "અમદાવાદનું હવામાન કેવું છે?" '
            "JSON સાધન કૉલ સાથે જવાબ આપો."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Ahmedabad"},
    },
    # ---- Punjabi ----
    "pa": {
        "prompt": (
            "ਤੁਹਾਡੇ ਕੋਲ ਇਹ ਟੂਲ ਹਨ: get_weather, search_web, translate। "
            'ਉਪਭੋਗਤਾ ਕਹਿੰਦਾ ਹੈ: "ਅੰਮ੍ਰਿਤਸਰ ਦਾ ਮੌਸਮ ਕਿਹੋ ਜਿਹਾ ਹੈ?" '
            "JSON ਟੂਲ ਕਾਲ ਨਾਲ ਜਵਾਬ ਦਿਓ।"
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Amritsar"},
    },
    # ---- Tamil ----
    "ta": {
        "prompt": (
            "உங்களிடம் இந்தக் கருவிகள் உள்ளன: get_weather, search_web, translate. "
            'பயனர் கூறுகிறார்: "சென்னையின் வானிலை என்ன?" '
            "JSON கருவி அழைப்பில் பதிலளிக்கவும்."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Chennai"},
    },
    # ---- Telugu ----
    "te": {
        "prompt": (
            "మీ వద్ద ఈ సాధనాలు ఉన్నాయి: get_weather, search_web, translate. "
            'వినియోగదారు చెప్పారు: "హైదరాబాద్ వాతావరణం ఎలా ఉంది?" '
            "JSON టూల్ కాల్\u200cతో సమాధానం ఇవ్వండి."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Hyderabad"},
    },
    # ---- Nepali ----
    "ne": {
        "prompt": (
            "तपाईंसँग यी उपकरणहरू छन्: get_weather, search_web, translate। "
            'प्रयोगकर्ता भन्छन्: "काठमाडौंको मौसम कस्तो छ?" '
            "JSON उपकरण कलबाट जवाफ दिनुहोस्।"
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Kathmandu"},
    },
    # ---- Filipino / Tagalog ----
    "tl": {
        "prompt": (
            "Mayroon kang mga tool na ito: get_weather, search_web, translate. "
            'Sinabi ng user: "Ano ang panahon sa Manila?" '
            "Sumagot gamit ang JSON tool call."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Manila"},
    },
    # ---- Malay ----
    "ms": {
        "prompt": (
            "Anda mempunyai alat ini: get_weather, search_web, translate. "
            'Pengguna berkata: "Bagaimana cuaca di Kuala Lumpur?" '
            "Jawab dengan panggilan alat JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Kuala Lumpur"},
    },
    # ---- Indonesian ----
    "id": {
        "prompt": (
            "Anda memiliki alat: get_weather, search_web, translate. "
            'Pengguna berkata: "Bagaimana cuaca di Jakarta?" '
            "Jawab dengan JSON tool call."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Jakarta"},
    },
    # ---- Vietnamese ----
    "vi": {
        "prompt": (
            "Bạn có các công cụ: get_weather, search_web, translate. "
            'Người dùng nói: "Thời tiết ở Hà Nội thế nào?" '
            "Trả lời bằng lệnh gọi công cụ JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Hanoi"},
    },
    # ---- Javanese ----
    "jv": {
        "prompt": (
            "Sampeyan duwe piranti iki: get_weather, search_web, translate. "
            'Pangguna kandha: "Kepiye cuacane ing Surabaya?" '
            "Wangsulana nganggo JSON tool call."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Surabaya"},
    },
    # ---- Khmer ----
    "km": {
        "prompt": (
            "អ្នកមានឧបករណ៍ទាំងនេះ: get_weather, search_web, translate។ "
            'អ្នកប្រើប្រាស់និយាយថា: "អាកាសធាតុនៅភ្នំពេញយ៉ាងម៉េច?" '
            "ឆ្លើយដោយប្រើការហៅឧបករណ៍ JSON។"
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Phnom Penh"},
    },
    # ---- Thai ----
    "th": {
        "prompt": (
            "คุณมีเครื่องมือเหล่านี้: get_weather, search_web, translate "
            'ผู้ใช้พูดว่า: "สภาพอากาศในกรุงเทพเป็นอย่างไร?" '
            "ตอบด้วยการเรียกเครื่องมือ JSON"
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Bangkok"},
    },
    # ---- Lao ----
    "lo": {
        "prompt": (
            "ທ່ານມີເຄື່ອງມືເຫຼົ່ານີ້: get_weather, search_web, translate. "
            'ຜູ້ໃຊ້ເວົ້າວ່າ: "ສະພາບອາກາດໃນວຽງຈັນເປັນແນວໃດ?" '
            "ຕອບດ້ວຍການເອີ້ນເຄື່ອງມື JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Vientiane"},
    },
    # ---- Burmese ----
    "my": {
        "prompt": (
            "သင့်တွင် ဤကိရိယာများရှိသည်: get_weather, search_web, translate။ "
            'အသုံးပြုသူက: "ရန်ကုန်ရဲ့ ရာသီဥတုဘယ်လိုလဲ?" ဟုပြောသည်။ '
            "JSON ကိရိယာခေါ်ဆိုမှုဖြင့် ဖြေပါ။"
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Yangon"},
    },
    # ---- Chinese ----
    "zh": {
        "prompt": (
            "你有这些工具: get_weather, search_web, translate。"
            '用户说："北京的天气怎么样？" '
            "用JSON工具调用回答。"
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Beijing"},
    },
    # ---- Japanese ----
    "ja": {
        "prompt": (
            "これらのツールがあります: get_weather, search_web, translate。"
            'ユーザーが言います：「東京の天気は？」'
            "JSONツールコールで応答してください。"
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Tokyo"},
    },
    # ---- Korean ----
    "ko": {
        "prompt": (
            "다음 도구가 있습니다: get_weather, search_web, translate. "
            '사용자가 말합니다: "서울 날씨가 어때요?" '
            "JSON 도구 호출로 응답하세요."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Seoul"},
    },
    # ---- Amharic ----
    "am": {
        "prompt": (
            "እነዚህ መሳሪያዎች አሉዎት: get_weather, search_web, translate። "
            'ተጠቃሚው ይላል: "በአዲስ አበባ የአየር ሁኔታ ምንድን ነው?" '
            "በJSON መሳሪያ ጥሪ ይመልሱ።"
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Addis Ababa"},
    },
    # ---- Hausa ----
    "ha": {
        "prompt": (
            "Kuna da waɗannan kayan aiki: get_weather, search_web, translate. "
            'Mai amfani ya ce: "Yaya yanayin yanayi a Kano?" '
            "Amsa da kiran kayan aiki na JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Kano"},
    },
    # ---- Igbo ----
    "ig": {
        "prompt": (
            "I nwere ngwaọrụ ndị a: get_weather, search_web, translate. "
            'Onye ọrụ na-ekwu: "Kedu ka ikuku dị na Enugu?" '
            "Za ngwa ọrụ JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Enugu"},
    },
    # ---- Malagasy ----
    "mg": {
        "prompt": (
            "Manana ireto fitaovana ireto ianao: get_weather, search_web, translate. "
            'Ny mpampiasa milaza hoe: "Manao ahoana ny toetrandro ao Antananarivo?" '
            "Valio amin'ny antso fitaovana JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Antananarivo"},
    },
    # ---- Shona ----
    "sn": {
        "prompt": (
            "Une maturusi aya: get_weather, search_web, translate. "
            'Mushandisi anoti: "Mamiriro ekunze muHarare akadii?" '
            "Pindura ne JSON tool call."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Harare"},
    },
    # ---- Swahili ----
    "sw": {
        "prompt": (
            "Una zana hizi: get_weather, search_web, translate. "
            'Mtumiaji anasema: "Hali ya hewa Nairobi ikoje?" '
            "Jibu kwa JSON tool call."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Nairobi"},
    },
    # ---- Wolof ----
    "wo": {
        "prompt": (
            "You have these tools: get_weather, search_web, translate. "
            'Jëfandikukat bi naan: "Nan la temps bi di nekk ci Dakar?" '
            "Tontu ak JSON tool call."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Dakar"},
    },
    # ---- Xhosa ----
    "xh": {
        "prompt": (
            "Unezi zixhobo: get_weather, search_web, translate. "
            'Umsebenzisi uthi: "Injani imozulu eKapa?" '
            "Phendula nge JSON tool call."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Cape Town"},
    },
    # ---- Yoruba ----
    "yo": {
        "prompt": (
            "O ni awọn irinṣẹ wọnyi: get_weather, search_web, translate. "
            'Olumulo sọ pe: "Bawo ni oju ojo ṣe ri ni Eko?" '
            "Dahun pẹlu ipe irinṣẹ JSON."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Lagos"},
    },
    # ---- Zulu ----
    "zu": {
        "prompt": (
            "Unamathuluzi alandelayo: get_weather, search_web, translate. "
            'Umsebenzisi uthi: "Sinjani isimo sezulu eThekwini?" '
            "Phendula nge JSON tool call."
        ),
        "expected_function": "get_weather",
        "expected_args": {"location": "Durban"},
    },
}


# ---------------------------------------------------------------------------
# Reasoning scenarios across four categories.
# Each category maps lang_code -> scenario with:
#   prompt: str — full prompt with CoT instruction
#   expected_answer: str — the correct final answer
#   answer_type: "numeric" | "choice" | "open"
#
# Representative subset of languages covering all 13 families.
# ---------------------------------------------------------------------------

_COT_INSTRUCTION = {
    "en": "Think step by step and show your reasoning before giving the final answer.",
    "es": "Piensa paso a paso y muestra tu razonamiento antes de dar la respuesta final.",
    "fr": "Réfléchis étape par étape et montre ton raisonnement avant de donner la réponse finale.",
    "de": "Denke Schritt für Schritt und zeige deine Überlegungen, bevor du die Antwort gibst.",
    "pt": "Pense passo a passo e mostre seu raciocínio antes de dar a resposta final.",
    "it": "Pensa passo dopo passo e mostra il tuo ragionamento prima di dare la risposta finale.",
    "nl": "Denk stap voor stap na en toon je redenering voordat je het antwoord geeft.",
    "pl": "Pomyśl krok po kroku i pokaż swoje rozumowanie przed podaniem odpowiedzi.",
    "ro": "Gândește pas cu pas și arată raționamentul tău înainte de a da răspunsul final.",
    "uk": "Думай крок за кроком і покажи своє міркування перед тим, як дати відповідь.",
    "cs": "Mysli krok za krokem a ukaž své uvažování, než dáš odpověď.",
    "el": "Σκέψου βήμα βήμα και δείξε τη συλλογιστική σου πριν δώσεις την απάντηση.",
    "hi": "कदम दर कदम सोचें और अंतिम उत्तर देने से पहले अपना तर्क दिखाएं।",
    "bn": "ধাপে ধাপে চিন্তা করুন এবং চূড়ান্ত উত্তর দেওয়ার আগে আপনার যুক্তি দেখান।",
    "mr": "टप्प्याटप्प्याने विचार करा आणि अंतिम उत्तर देण्यापूर्वी तुमचे तर्क दाखवा.",
    "gu": "પગલે પગલે વિચારો અને અંતિમ જવાબ આપતા પહેલા તમારું તર્ક બતાવો.",
    "ne": "चरणबद्ध रूपमा सोच्नुहोस् र अन्तिम उत्तर दिनुअघि आफ्नो तर्क देखाउनुहोस्।",
    "ta": "படிப்படியாக சிந்தித்து இறுதி பதிலை அளிப்பதற்கு முன் உங்கள் தர்க்கத்தைக் காட்டுங்கள்.",
    "te": "అడుగు అడుగున ఆలోచించండి మరియు తుది సమాధానం ఇవ్వడానికి ముందు మీ తర్కం చూపించండి.",
    "ar": "فكر خطوة بخطوة وأظهر تفكيرك قبل إعطاء الإجابة النهائية.",
    "fa": "قدم به قدم فکر کنید و استدلال خود را قبل از پاسخ نهایی نشان دهید.",
    "he": "חשוב צעד אחר צעד והראה את ההיגיון שלך לפני שאתה נותן את התשובה הסופית.",
    "tr": "Adım adım düşün ve son cevabı vermeden önce akıl yürütmeni göster.",
    "az": "Addım-addım düşünün və son cavabı verməzdən əvvəl mühakimənizi göstərin.",
    "uz": "Qadam-baqadam o'ylang va yakuniy javobni berishdan oldin fikringizni ko'rsating.",
    "kk": "Қадам-қадаммен ойланыңыз және соңғы жауапты бермес бұрын пайымдауыңызды көрсетіңіз.",
    "zh": "请一步一步思考，在给出最终答案之前展示你的推理过程。",
    "ja": "ステップバイステップで考え、最終的な答えを出す前に推論を示してください。",
    "ko": "단계별로 생각하고 최종 답을 내리기 전에 추론을 보여주세요.",
    "sw": "Fikiria hatua kwa hatua na uonyeshe mantiki yako kabla ya kutoa jibu la mwisho.",
    "yo": "Ronu igbesẹ nipasẹ igbesẹ ki o si fi ironu rẹ han ṣaaju ki o to fun ni idahun ikẹhin.",
    "ig": "Chee echiche nzọụkwụ site na nzọụkwụ ma gosipụta ezi uche gị tupu ịnye azịza ikpeazụ.",
    "ha": "Yi tunani mataki-mataki kuma nuna tunaninki kafin bayar da amsar ƙarshe.",
    "zu": "Cabanga isinyathelo ngesinyathelo futhi ukhombise isizathu sakho ngaphambi kokunikeza impendulo yokugcina.",
    "id": "Pikirkan langkah demi langkah dan tunjukkan penalaran Anda sebelum memberikan jawaban akhir.",
    "ms": "Fikir langkah demi langkah dan tunjukkan penaakulan anda sebelum memberi jawapan akhir.",
    "tl": "Mag-isip nang hakbang-hakbang at ipakita ang iyong pangangatwiran bago ibigay ang panghuling sagot.",
    "vi": "Hãy suy nghĩ từng bước và trình bày lập luận trước khi đưa ra câu trả lời cuối cùng.",
    "th": "คิดทีละขั้นตอนและแสดงเหตุผลก่อนให้คำตอบสุดท้าย",
    "km": "គិតជាជំហានៗ ហើយបង្ហាញហេតុផលរបស់អ្នកមុនពេលផ្តល់ចម្លើយចុងក្រោយ។",
    "lo": "ຄິດເທື່ອລະຂັ້ນ ແລະ ສະແດງເຫດຜົນຂອງທ່ານກ່ອນຈະໃຫ້ຄຳຕອບສຸດທ້າຍ.",
    "my": "အဆင့်ဆင့် စဥ်းစားပြီး နောက်ဆုံးအဖြေမပေးမီ သင့်ကျိုးကြောင်းဆင်ခြင်ကို ပြပါ။",
    "fi": "Ajattele askel askeleelta ja näytä päättelysi ennen lopullisen vastauksen antamista.",
    "hu": "Gondolkodj lépésről lépésre, és mutasd meg az érvelésedet, mielőtt megadnád a választ.",
    "et": "Mõtle samm-sammult ja näita oma arutluskäiku enne lõpliku vastuse andmist.",
    "ka": "იფიქრეთ ნაბიჯ-ნაბიჯ და აჩვენეთ თქვენი მსჯელობა საბოლოო პასუხის გაცემამდე.",
    "am": "ደረጃ በደረጃ አስቡ እና የመጨረሻውን መልስ ከመስጠትዎ በፊት ምክንያትዎን ያሳዩ።",
    "mg": "Eritrereto dingana isaky ny dingana ary asehoy ny fanaparitahanao alohan'ny fanomezana ny valiny farany.",
    "sn": "Funga nhanho nhanho uye ratidza mafungiro ako usati wapa mhinduro yekupedzisira.",
}


def _math_prompt(lang: str, cot: str) -> str:
    """Build a mathematical reasoning prompt with CoT."""
    problems = {
        "en": "A store sells apples for $2 each and oranges for $3 each. Maria buys 4 apples and 5 oranges. How much does she spend in total? {cot}",
        "es": "Una tienda vende manzanas a $2 cada una y naranjas a $3 cada una. María compra 4 manzanas y 5 naranjas. ¿Cuánto gasta en total? {cot}",
        "fr": "Un magasin vend des pommes à 2$ pièce et des oranges à 3$ pièce. Maria achète 4 pommes et 5 oranges. Combien dépense-t-elle au total ? {cot}",
        "de": "Ein Laden verkauft Äpfel für 2$ und Orangen für 3$ das Stück. Maria kauft 4 Äpfel und 5 Orangen. Wie viel gibt sie insgesamt aus? {cot}",
        "hi": "एक दुकान $2 प्रति सेब और $3 प्रति संतरे बेचती है। मारिया 4 सेब और 5 संतरे खरीदती है। कुल कितना खर्च करती है? {cot}",
        "ar": "يبيع متجر التفاح بـ 2 دولار والبرتقال بـ 3 دولارات. اشترت ماريا 4 تفاحات و5 برتقالات. كم أنفقت إجمالاً؟ {cot}",
        "zh": "一家商店苹果每个2美元，橙子每个3美元。玛丽亚买了4个苹果和5个橙子。她总共花了多少钱？{cot}",
        "ja": "ある店ではリンゴが1個2ドル、オレンジが1個3ドルです。マリアはリンゴを4個とオレンジを5個買いました。合計いくらですか？{cot}",
        "ko": "한 가게에서 사과는 개당 2달러, 오렌지는 개당 3달러에 팝니다. 마리아가 사과 4개와 오렌지 5개를 샀습니다. 총 얼마를 썼나요? {cot}",
        "tr": "Bir mağaza elmaları 2$, portakalları 3$ karşılığında satıyor. Maria 4 elma ve 5 portakal alıyor. Toplamda ne kadar harcıyor? {cot}",
        "sw": "Duka linauza tufaha kwa $2 kila moja na machungwa kwa $3 kila moja. Maria ananunua tufaha 4 na machungwa 5. Anatumia kiasi gani jumla? {cot}",
        "id": "Sebuah toko menjual apel seharga $2 per buah dan jeruk seharga $3 per buah. Maria membeli 4 apel dan 5 jeruk. Berapa total yang dia belanjakan? {cot}",
        "vi": "Một cửa hàng bán táo với giá 2 đô la mỗi quả và cam với giá 3 đô la mỗi quả. Maria mua 4 quả táo và 5 quả cam. Tổng cộng cô ấy chi bao nhiêu? {cot}",
        "th": "ร้านค้าขายแอปเปิ้ลราคาลูกละ 2 ดอลลาร์และส้มราคาลูกละ 3 ดอลลาร์ มาเรียซื้อแอปเปิ้ล 4 ลูกและส้ม 5 ลูก เธอใช้จ่ายทั้งหมดเท่าไร? {cot}",
        "bn": "একটি দোকান প্রতিটি আপেল $2 এবং প্রতিটি কমলা $3 তে বিক্রি করে। মারিয়া 4টি আপেল এবং 5টি কমলা কেনে। সে মোট কত খরচ করে? {cot}",
        "te": "ఒక దుకాణం ఆపిల్‌ను ఒక్కొక్కటి $2 కు మరియు నారింజను ఒక్కొక్కటి $3 కు అమ్ముతుంది. మారియా 4 ఆపిల్స్ మరియు 5 నారింజలు కొంటుంది. ఆమె మొత్తం ఎంత ఖర్చు చేసింది? {cot}",
        "ta": "ஒரு கடை ஒரு ஆப்பிள் $2 க்கும் ஒரு ஆரஞ்சு $3 க்கும் விற்கிறது. மரியா 4 ஆப்பிள்களும் 5 ஆரஞ்சுகளும் வாங்குகிறாள். மொத்தம் எவ்வளவு செலவழிக்கிறாள்? {cot}",
        "fi": "Kauppa myy omenoita 2$ kappale ja appelsiineja 3$ kappale. Maria ostaa 4 omenaa ja 5 appelsiinia. Kuinka paljon hän käyttää yhteensä? {cot}",
        "hu": "Egy bolt 2$-ért árul almát és 3$-ért narancsot. Maria 4 almát és 5 narancsot vesz. Összesen mennyit költ? {cot}",
        "pl": "Sklep sprzedaje jabłka po 2$ i pomarańcze po 3$ za sztukę. Maria kupuje 4 jabłka i 5 pomarańczy. Ile wydaje łącznie? {cot}",
        "uk": "Магазин продає яблука по $2 та апельсини по $3 за штуку. Марія купує 4 яблука та 5 апельсинів. Скільки вона витрачає? {cot}",
        "fa": "یک فروشگاه هر سیب را 2 دلار و هر پرتقال را 3 دلار می‌فروشد. ماریا 4 سیب و 5 پرتقال می‌خرد. در مجموع چقدر خرج می‌کند؟ {cot}",
        "yo": "Ile itaja kan n ta apple fun $2 kọọkan ati osan fun $3 kọọkan. Maria ra apple 4 ati osan 5. Iye melo ni o na lapapọ? {cot}",
        "ig": "Ụlọ ahịa na-ere apụl $2 otu otu na oroma $3 otu otu. Maria zụtara apụl 4 na oroma 5. Ego ole ka o tụrụ niile? {cot}",
        "ha": "Shagon yana sayar da apple a $2 kowannensu da lemu a $3. Maria ta sayi apple 4 da lemu 5. Nawa ta kashe baki daya? {cot}",
        "zu": "Isitolo sithengisa ama-apula nge-$2 ngalinye nama-orenji nge-$3 ngalinye. UMaria uthenga ama-apula angu-4 nama-orenji angu-5. Usebenzise malini esewonke? {cot}",
        "am": "አንድ ሱቅ ፖም በ$2 እና ብርቱካን በ$3 ይሸጣል። ማሪያ 4 ፖም እና 5 ብርቱካን ትገዛለች። በጠቅላላ ስንት ወጪ ታደርጋለች? {cot}",
        "ka": "მაღაზია ყიდის ვაშლს 2$-ად და ფორთოხალს 3$-ად. მარია ყიდულობს 4 ვაშლს და 5 ფორთოხალს. სულ რამდენს ხარჯავს? {cot}",
        "km": "ហាងលក់ផ្លែប៉ោម $2 មួយ និងក្រូច $3 មួយ។ ម៉ារីយ៉ាទិញផ្លែប៉ោម 4 និងក្រូច 5។ តើនាងចំណាយប៉ុន្មានសរុប? {cot}",
        "my": "ဆိုင်တစ်ဆိုင်က ပန်းသီးတစ်လုံး $2 နဲ့ လိမ္မော်သီးတစ်လုံး $3 ရောင်းတယ်။ မာရီယာက ပန်းသီး 4 လုံးနဲ့ လိမ္မော်သီး 5 လုံး ဝယ်တယ်။ စုစုပေါင်း ဘယ်လောက်ကုန်သလဲ? {cot}",
        "mg": "Fivarotana iray mivarotra paoma $2 tsirairay sy voasary $3 tsirairay. Maria mividy paoma 4 sy voasary 5. Ohatrinona no laniny amin'ny fitambarany? {cot}",
        "sn": "Chitoro chinotengesa maapuro ne$2 rimwe nerimwe uye maorange ne$3. Maria atenga maapuro 4 neorange 5. Akashandisa marii yese? {cot}",
    }
    template = problems.get(lang, problems["en"])
    return template.format(cot=cot)


def _causal_prompt(lang: str, cot: str) -> str:
    """Build a causal reasoning prompt with CoT."""
    problems = {
        "en": "The road was wet. What is the most likely cause? (A) It rained (B) It was sunny (C) It was windy (D) It was cold. {cot}",
        "es": "La carretera estaba mojada. ¿Cuál es la causa más probable? (A) Llovió (B) Hacía sol (C) Hacía viento (D) Hacía frío. {cot}",
        "fr": "La route était mouillée. Quelle est la cause la plus probable ? (A) Il a plu (B) Il faisait soleil (C) Il y avait du vent (D) Il faisait froid. {cot}",
        "de": "Die Straße war nass. Was ist die wahrscheinlichste Ursache? (A) Es hat geregnet (B) Es war sonnig (C) Es war windig (D) Es war kalt. {cot}",
        "hi": "सड़क गीली थी। सबसे संभावित कारण क्या है? (A) बारिश हुई (B) धूप थी (C) हवा चल रही थी (D) ठंड थी। {cot}",
        "ar": "كان الطريق مبللاً. ما السبب الأكثر احتمالاً؟ (أ) هطل المطر (ب) كان الجو مشمساً (ج) كان الجو عاصفاً (د) كان الجو بارداً. {cot}",
        "zh": "路面是湿的。最可能的原因是什么？(A) 下雨了 (B) 天晴 (C) 刮风 (D) 天冷。{cot}",
        "ja": "道路が濡れていました。最も可能性の高い原因は何ですか？(A) 雨が降った (B) 晴れていた (C) 風が強かった (D) 寒かった。{cot}",
        "ko": "도로가 젖어 있었습니다. 가장 가능성 있는 원인은? (A) 비가 왔다 (B) 맑았다 (C) 바람이 불었다 (D) 추웠다. {cot}",
        "tr": "Yol ıslaktı. En muhtemel neden nedir? (A) Yağmur yağdı (B) Güneşliydi (C) Rüzgârlıydı (D) Soğuktu. {cot}",
        "sw": "Barabara ilikuwa na unyevu. Nini sababu inayowezekana zaidi? (A) Mvua ilinyesha (B) Kulikuwa na jua (C) Kulikuwa na upepo (D) Kulikuwa baridi. {cot}",
        "id": "Jalan basah. Apa penyebab yang paling mungkin? (A) Hujan turun (B) Cerah (C) Berangin (D) Dingin. {cot}",
        "vi": "Đường ướt. Nguyên nhân có khả năng nhất là gì? (A) Trời mưa (B) Trời nắng (C) Trời gió (D) Trời lạnh. {cot}",
        "th": "ถนนเปียก สาเหตุที่เป็นไปได้มากที่สุดคืออะไร? (A) ฝนตก (B) แดดออก (C) ลมแรง (D) อากาศเย็น {cot}",
        "bn": "রাস্তা ভেজা ছিল। সবচেয়ে সম্ভাব্য কারণ কী? (A) বৃষ্টি হয়েছিল (B) রোদ ছিল (C) বাতাস ছিল (D) ঠান্ডা ছিল। {cot}",
        "te": "రోడ్డు తడిగా ఉంది. అత్యంత సంభావ్య కారణం ఏమిటి? (A) వర్షం పడింది (B) ఎండగా ఉంది (C) గాలి వీచింది (D) చలిగా ఉంది. {cot}",
        "ta": "சாலை ஈரமாக இருந்தது. மிகவும் சாத்தியமான காரணம் என்ன? (A) மழை பெய்தது (B) வெயில் அடித்தது (C) காற்று வீசியது (D) குளிராக இருந்தது. {cot}",
        "fi": "Tie oli märkä. Mikä on todennäköisin syy? (A) Satoi (B) Paistoi aurinko (C) Tuuli (D) Kylmää. {cot}",
        "hu": "Az út nedves volt. Mi a legvalószínűbb ok? (A) Esett az eső (B) Napsütés (C) Szél (D) Hideg. {cot}",
        "pl": "Droga była mokra. Jaka jest najbardziej prawdopodobna przyczyna? (A) Padał deszcz (B) Było słonecznie (C) Wiało (D) Było zimno. {cot}",
        "uk": "Дорога була мокрою. Яка найбільш імовірна причина? (A) Був дощ (B) Було сонячно (C) Був вітер (D) Було холодно. {cot}",
    }
    template = problems.get(lang, problems["en"])
    return template.format(cot=cot)


def _analogical_prompt(lang: str, cot: str) -> str:
    """Build an analogical reasoning prompt with CoT."""
    problems = {
        "en": "Hot is to cold as day is to ___. (A) night (B) morning (C) sun (D) warm. {cot}",
        "es": "Caliente es a frío como día es a ___. (A) noche (B) mañana (C) sol (D) cálido. {cot}",
        "fr": "Chaud est à froid comme jour est à ___. (A) nuit (B) matin (C) soleil (D) tiède. {cot}",
        "de": "Heiß verhält sich zu kalt wie Tag zu ___. (A) Nacht (B) Morgen (C) Sonne (D) warm. {cot}",
        "hi": "गर्म का ठंडा से वही संबंध है जो दिन का ___ से है। (A) रात (B) सुबह (C) सूरज (D) गर्म। {cot}",
        "ar": "الحار للبارد كالنهار لـ ___. (أ) الليل (ب) الصباح (ج) الشمس (د) الدفء. {cot}",
        "zh": "热之于冷，如同白天之于___。(A) 夜晚 (B) 早晨 (C) 太阳 (D) 温暖。{cot}",
        "ja": "暑いと寒いの関係は、昼と___の関係と同じです。(A) 夜 (B) 朝 (C) 太陽 (D) 暖かい。{cot}",
        "ko": "뜨거움이 차가움에 대한 것처럼, 낮은 ___에 대한 것입니다. (A) 밤 (B) 아침 (C) 태양 (D) 따뜻함. {cot}",
        "tr": "Sıcak soğuğa ne ise gündüz ___'e odur. (A) gece (B) sabah (C) güneş (D) ılık. {cot}",
        "sw": "Moto kwa baridi ni sawa na mchana kwa ___. (A) usiku (B) asubuhi (C) jua (D) joto. {cot}",
        "id": "Panas berbanding dingin seperti siang berbanding ___. (A) malam (B) pagi (C) matahari (D) hangat. {cot}",
        "vi": "Nóng đối với lạnh giống như ngày đối với ___. (A) đêm (B) sáng (C) mặt trời (D) ấm. {cot}",
        "th": "ร้อนต่อเย็น เหมือนกลางวันต่อ ___. (A) กลางคืน (B) เช้า (C) ดวงอาทิตย์ (D) อบอุ่น {cot}",
        "bn": "গরম যেমন ঠান্ডার সাথে সম্পর্কিত, দিন তেমন ___-এর সাথে। (A) রাত (B) সকাল (C) সূর্য (D) উষ্ণ। {cot}",
        "te": "వేడి చలికి ఎలాగో పగలు ___కి అలాగే. (A) రాత్రి (B) ఉదయం (C) సూర్యుడు (D) వెచ్చగా. {cot}",
        "ta": "சூடு குளிருக்கு எப்படியோ பகல் ___க்கு அப்படியே. (A) இரவு (B) காலை (C) சூரியன் (D) சூடான. {cot}",
        "fi": "Kuuma on kylmälle kuin päivä on ___. (A) yö (B) aamu (C) aurinko (D) lämmin. {cot}",
        "hu": "A meleg úgy viszonyul a hideghez, mint a nappal a ___. (A) éjszaka (B) reggel (C) nap (D) meleg. {cot}",
        "pl": "Gorąco jest do zimna jak dzień do ___. (A) noc (B) rano (C) słońce (D) ciepło. {cot}",
        "uk": "Гаряче до холодного як день до ___. (A) ніч (B) ранок (C) сонце (D) тепло. {cot}",
    }
    template = problems.get(lang, problems["en"])
    return template.format(cot=cot)


def _logical_prompt(lang: str, cot: str) -> str:
    """Build a logical reasoning prompt with CoT."""
    problems = {
        "en": "All birds can fly. Penguins are birds. Can penguins fly? (A) Yes (B) No (C) Sometimes (D) Only when young. Note: answer based ONLY on the given premises. {cot}",
        "es": "Todos los pájaros pueden volar. Los pingüinos son pájaros. ¿Pueden volar los pingüinos? (A) Sí (B) No (C) A veces (D) Solo cuando son jóvenes. Nota: responda SOLO basándose en las premisas dadas. {cot}",
        "fr": "Tous les oiseaux peuvent voler. Les pingouins sont des oiseaux. Les pingouins peuvent-ils voler ? (A) Oui (B) Non (C) Parfois (D) Seulement quand ils sont jeunes. Note : répondez UNIQUEMENT sur la base des prémisses données. {cot}",
        "de": "Alle Vögel können fliegen. Pinguine sind Vögel. Können Pinguine fliegen? (A) Ja (B) Nein (C) Manchmal (D) Nur wenn jung. Hinweis: Antworten Sie NUR basierend auf den gegebenen Prämissen. {cot}",
        "hi": "सभी पक्षी उड़ सकते हैं। पेंगुइन पक्षी हैं। क्या पेंगुइन उड़ सकते हैं? (A) हाँ (B) नहीं (C) कभी-कभी (D) केवल जब छोटे हों। नोट: केवल दिए गए आधार पर उत्तर दें। {cot}",
        "ar": "كل الطيور تستطيع الطيران. البطاريق طيور. هل تستطيع البطاريق الطيران؟ (أ) نعم (ب) لا (ج) أحياناً (د) فقط عندما تكون صغيرة. ملاحظة: أجب بناءً على المقدمات المعطاة فقط. {cot}",
        "zh": "所有鸟都会飞。企鹅是鸟。企鹅会飞吗？(A) 是 (B) 否 (C) 有时 (D) 只在年幼时。注意：仅根据给定的前提回答。{cot}",
        "ja": "すべての鳥は飛べます。ペンギンは鳥です。ペンギンは飛べますか？(A) はい (B) いいえ (C) 時々 (D) 若い時だけ。注意：与えられた前提のみに基づいて答えてください。{cot}",
        "ko": "모든 새는 날 수 있습니다. 펭귄은 새입니다. 펭귄은 날 수 있나요? (A) 예 (B) 아니오 (C) 가끔 (D) 어릴 때만. 참고: 주어진 전제만을 기반으로 답하세요. {cot}",
        "tr": "Bütün kuşlar uçabilir. Penguenler kuştur. Penguenler uçabilir mi? (A) Evet (B) Hayır (C) Bazen (D) Sadece gençken. Not: SADECE verilen öncüllere göre cevaplayın. {cot}",
        "sw": "Ndege wote wanaweza kuruka. Pengwini ni ndege. Je, pengwini wanaweza kuruka? (A) Ndiyo (B) Hapana (C) Wakati mwingine (D) Wakiwa wadogo tu. Kumbuka: jibu TU kulingana na maelezo yaliyotolewa. {cot}",
        "id": "Semua burung bisa terbang. Penguin adalah burung. Bisakah penguin terbang? (A) Ya (B) Tidak (C) Kadang-kadang (D) Hanya saat muda. Catatan: jawab HANYA berdasarkan premis yang diberikan. {cot}",
        "vi": "Tất cả chim đều biết bay. Chim cánh cụt là chim. Chim cánh cụt có biết bay không? (A) Có (B) Không (C) Đôi khi (D) Chỉ khi còn nhỏ. Lưu ý: chỉ trả lời dựa trên tiền đề đã cho. {cot}",
        "th": "นกทุกตัวบินได้ เพนกวินเป็นนก เพนกวินบินได้ไหม? (A) ใช่ (B) ไม่ (C) บางครั้ง (D) เฉพาะตอนยังเล็ก หมายเหตุ: ตอบตามสมมติฐานที่ให้มาเท่านั้น {cot}",
        "bn": "সব পাখি উড়তে পারে। পেঙ্গুইন পাখি। পেঙ্গুইন কি উড়তে পারে? (A) হ্যাঁ (B) না (C) মাঝে মাঝে (D) শুধু ছোটবেলায়। দ্রষ্টব্য: শুধুমাত্র প্রদত্ত ভিত্তির উপর ভিত্তি করে উত্তর দিন। {cot}",
        "te": "అన్ని పక్షులు ఎగురగలవు. పెంగ్విన్‌లు పక్షులు. పెంగ్విన్‌లు ఎగురగలవా? (A) అవును (B) కాదు (C) కొన్నిసార్లు (D) చిన్నప్పుడు మాత్రమే. గమనిక: ఇచ్చిన ఆధారాల ఆధారంగా మాత్రమే సమాధానం ఇవ్వండి. {cot}",
        "ta": "எல்லா பறவைகளும் பறக்கும். பெங்குயின்கள் பறவைகள். பெங்குயின்கள் பறக்க முடியுமா? (A) ஆம் (B) இல்லை (C) சில நேரங்களில் (D) இளமையாக இருக்கும்போது மட்டும். குறிப்பு: கொடுக்கப்பட்ட முன்மொழிவுகளின் அடிப்படையில் மட்டுமே பதிலளிக்கவும். {cot}",
        "fi": "Kaikki linnut voivat lentää. Pingviinit ovat lintuja. Voivatko pingviinit lentää? (A) Kyllä (B) Ei (C) Joskus (D) Vain nuorina. Huomio: vastaa VAIN annettujen premissien perusteella. {cot}",
        "hu": "Minden madár tud repülni. A pingvinek madarak. Tudnak-e a pingvinek repülni? (A) Igen (B) Nem (C) Néha (D) Csak fiatalon. Megjegyzés: CSAK a megadott premisszák alapján válaszoljon. {cot}",
        "pl": "Wszystkie ptaki potrafią latać. Pingwiny to ptaki. Czy pingwiny potrafią latać? (A) Tak (B) Nie (C) Czasami (D) Tylko gdy są młode. Uwaga: odpowiedz WYŁĄCZNIE na podstawie podanych przesłanek. {cot}",
        "uk": "Усі птахи вміють літати. Пінгвіни — це птахи. Чи можуть пінгвіни літати? (A) Так (B) Ні (C) Іноді (D) Тільки коли молоді. Примітка: відповідайте ЛИШЕ на основі наданих передумов. {cot}",
    }
    template = problems.get(lang, problems["en"])
    return template.format(cot=cot)


def _build_reasoning_scenarios() -> dict[str, dict[str, dict[str, Any]]]:
    """Build reasoning scenarios for all languages with CoT instructions."""
    scenarios: dict[str, dict[str, dict[str, Any]]] = {
        "mathematical": {},
        "causal": {},
        "analogical": {},
        "logical": {},
    }

    # Get all languages that have CoT instructions
    for lang in _COT_INSTRUCTION:
        cot = _COT_INSTRUCTION[lang]

        # Mathematical: 4*2 + 5*3 = 8 + 15 = 23
        scenarios["mathematical"][lang] = {
            "prompt": _math_prompt(lang, cot),
            "expected_answer": "23",
            "answer_type": "numeric",
        }

        # Causal: wet road -> rain
        scenarios["causal"][lang] = {
            "prompt": _causal_prompt(lang, cot),
            "expected_answer": "A",
            "answer_type": "choice",
        }

        # Analogical: hot:cold :: day:night
        scenarios["analogical"][lang] = {
            "prompt": _analogical_prompt(lang, cot),
            "expected_answer": "A",
            "answer_type": "choice",
        }

        # Logical: all birds fly + penguins are birds -> yes (from premises)
        scenarios["logical"][lang] = {
            "prompt": _logical_prompt(lang, cot),
            "expected_answer": "A",
            "answer_type": "choice",
        }

    return scenarios


REASONING_SCENARIOS: dict[str, dict[str, dict[str, Any]]] = _build_reasoning_scenarios()
