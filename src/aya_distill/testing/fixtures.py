"""Test fixtures and sample data for multilingual testing.

Covers all 70+ languages from CohereLabs/tiny-aya-global.
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
