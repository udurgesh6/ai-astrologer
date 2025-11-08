"""
Comprehensive Domain Definitions for Astrological Knowledge Bases
Includes keywords in English, Sanskrit, Hindi, and Marathi
"""

# Complete domain definitions with extensive multilingual keywords
DOMAIN_DEFINITIONS = {
    'base': {
        'keywords': [
            # English
            'vedic astrology', 'elements', 'general', 'basics', 'fundamental',
            'astrology', 'horoscope', 'birth chart', 'natal chart', 'kundali',
            'planets', 'houses', 'signs', 'zodiac', 'ascendant', 'lagna',
            
            # Sanskrit
            'jyotish', 'jyotisha', 'hora', 'hora shastra', 'brihat parashara',
            'shastra', 'graha', 'rashi', 'bhava', 'kendra', 'trikona',
            
            # Hindi
            'kundli', 'janam kundali', 'janampatri', 'varsha phal',
            
            # Marathi
            'jyotish shastra', 'janam kundali'
        ],
        'description': 'Base vedic astrology knowledge - always included',
        'always_include': True
    },
    
    'children': {
        'keywords': [
            # English - General
            'children', 'child', 'kids', 'offspring', 'progeny', 'son', 'daughter',
            'baby', 'infant', 'toddler', 'childbirth', 'pregnancy', 'conceive',
            'conception', 'fertility', 'infertility', 'miscarriage', 'abortion',
            
            # English - Specific terms
            'firstborn', 'second child', 'twins', 'multiple births',
            'adoption', 'stepchild', 'foster child',
            'child health', 'child education', 'child marriage',
            
            # Sanskrit
            'putra', 'putri', 'santana', 'santan', 'aputra', 'niraputra',
            'putra bhava', 'pancham bhava', 'pancham sthan',
            'putra yoga', 'santana yoga', 'aputra yoga',
            'putra karaka', 'jupiter', 'guru',
            
            # Hindi
            'santaan', 'beta', 'beti', 'baccha', 'bache', 'aulad',
            'santaan prapti', 'santaan sukh', 'santaan dosh',
            'putr', 'putri', 'panchva bhav', '5va ghar',
            
            # Marathi
            'mool', 'mulagaa', 'mulgi', 'santati', 'potgayi',
            'panchama bhav'
        ],
        'description': 'Children, progeny, and fertility related matters',
        'always_include': False
    },
    
    'marriage': {
        'keywords': [
            # English - General
            'marriage', 'married', 'marry', 'wedding', 'spouse', 'partner',
            'husband', 'wife', 'bride', 'groom', 'matrimony',
            'engagement', 'betrothal', 'wedding ceremony',
            
            # English - Relationship types
            'love marriage', 'arranged marriage', 'inter-caste marriage',
            'second marriage', 'remarriage', 'late marriage', 'early marriage',
            'delayed marriage', 'marriage timing', 'marriage age',
            
            # English - Relationship issues
            'relationship', 'romance', 'love', 'affair', 'dating',
            'compatibility', 'matching', 'divorce', 'separation',
            'marital discord', 'marital problems', 'marital happiness',
            'extramarital', 'adultery', 'infidelity',
            
            # English - Partner qualities
            'spouse nature', 'spouse appearance', 'spouse profession',
            'spouse character', 'foreign spouse', 'wealthy spouse',
            
            # Sanskrit
            'vivaha', 'vivah', 'pani-grahana', 'kalyana',
            'kalatra', 'dampati', 'pati', 'patni', 'bharya', 'bhartru',
            'saptam bhava', 'saptama bhava', 'kalatra bhava',
            'saptam sthan', 'seventh house',
            'vivaha yoga', 'kalatra yoga', 'marriage yoga',
            'vivaha karaka', 'venus', 'shukra',
            'upapada lagna', 'darakaraka',
            
            # Hindi
            'shaadi', 'shadi', 'byah', 'biyah', 'vivah',
            'pati', 'patni', 'dulha', 'dulhan',
            'suhag', 'suhaagan', 'suhagan',
            'saatva ghar', '7va bhav', 'saptam bhav',
            'prem vivah', 'arrange marriage',
            'vivah yog', 'shaadi ka time', 'shaadi kab hogi',
            'dusra vivah', 'talaq', 'vivah vichhed',
            
            # Marathi
            'lagn', 'vivah', 'pati', 'patni', 'var', 'vadhu',
            'saptam bhav', 'kalatra bhav',
            'prem vivah', 'talak', 'ghatak'
        ],
        'description': 'Marriage, relationships, and spouse related matters',
        'always_include': False
    },
    
    'career': {
        'keywords': [
            # English - General
            'career', 'job', 'profession', 'work', 'employment', 'occupation',
            'vocation', 'calling', 'livelihood', 'business', 'trade',
            'service', 'government job', 'private job', 'self-employment',
            
            # English - Job aspects
            'job change', 'job loss', 'unemployment', 'promotion',
            'transfer', 'posting', 'salary', 'increment', 'raise',
            'success', 'failure', 'achievement', 'recognition',
            'competition', 'competitive exam',
            
            # English - Business
            'entrepreneurship', 'startup', 'business success', 'business failure',
            'partnership', 'company', 'enterprise', 'venture',
            'profit', 'loss', 'bankruptcy',
            
            # English - Career fields
            'engineering', 'medicine', 'law', 'teaching', 'politics',
            'army', 'military', 'police', 'civil service', 'ias', 'ips',
            'banking', 'finance', 'accounting', 'management',
            'it', 'software', 'technology', 'research',
            'arts', 'media', 'journalism', 'acting', 'sports',
            
            # Sanskrit
            'karma', 'karma bhava', 'dasham bhava', 'dasham sthan',
            'tenth house', '10th house',
            'karma karaka', 'saturn', 'shani',
            'vyapara', 'seva', 'rajya seva',
            'karma yoga', 'profession yoga',
            
            # Hindi
            'naukri', 'job', 'vyapar', 'vyavasay', 'karobar', 'dhandha',
            'sarkari naukri', 'private naukri', 'business',
            'dasva ghar', '10va bhav', 'karm bhav',
            'teraki', 'padonnati', 'promotion',
            'naukri badalna', 'berojgari', 'naukri chod dena',
            'safalta', 'asafalta', 'kamyabi',
            
            # Marathi
            'nokri', 'vyavasay', 'udyog', 'vyapar', 'dhanda',
            'shaskiya nokri', 'sarkari nokri',
            'dasham bhav', 'karma bhav'
        ],
        'description': 'Career, profession, business, and work related matters',
        'always_include': False
    },
    
    'wealth': {
        'keywords': [
            # English - General
            'wealth', 'money', 'finance', 'finances', 'financial',
            'riches', 'prosperity', 'fortune', 'affluence',
            'income', 'earnings', 'salary', 'wages', 'revenue',
            'assets', 'property', 'estate', 'real estate', 'land',
            
            # English - Money matters
            'savings', 'investment', 'stocks', 'shares', 'mutual funds',
            'gold', 'silver', 'jewellery', 'jewelry',
            'bank balance', 'fixed deposit', 'fd',
            'profit', 'loss', 'gain', 'expenditure', 'expense',
            
            # English - Debt and loans
            'debt', 'loan', 'credit', 'borrowing', 'lending',
            'emi', 'mortgage', 'home loan', 'personal loan',
            'creditor', 'debtor', 'bankruptcy', 'insolvency',
            
            # English - Wealth sources
            'inheritance', 'ancestral property', 'legacy',
            'lottery', 'gambling', 'windfall', 'sudden gain',
            'speculation', 'trading', 'business profit',
            
            # English - Houses
            '2nd house', 'second house', 'house of wealth',
            '11th house', 'eleventh house', 'house of gains',
            '8th house', 'eighth house', 'hidden wealth', 'sudden wealth',
            
            # Sanskrit
            'dhana', 'artha', 'sampatti', 'sampada', 'aishwarya',
            'dwitiya bhava', 'dhana bhava', 'second house',
            'ekadash bhava', 'labha bhava', 'eleventh house',
            'ashtam bhava', 'eighth house',
            'dhana yoga', 'lakshmi yoga', 'wealth yoga',
            'dhana karaka', 'jupiter', 'guru', 'mercury', 'budh',
            'runa', 'karz', 'debt',
            
            # Hindi
            'dhan', 'paisa', 'sampatti', 'daulat', 'daular',
            'aay', 'aamdani', 'kamai', 'income',
            'bachat', 'savings', 'nivesh', 'investment',
            'labh', 'faida', 'munafa', 'profit',
            'nuksan', 'loss', 'ghaata',
            'dusra ghar', '2nd bhav', 'dhana bhav',
            'gyarva ghar', '11va bhav', 'labh bhav',
            'aathva ghar', '8va bhav',
            'karz', 'udhaar', 'loan', 'debt',
            'virasat', 'purkha dhan', 'inheritance',
            'lottery', 'jua', 'satta', 'gambling',
            
            # Marathi
            'dhan', 'sampatti', 'paisa', 'daular',
            'utpanna', 'income', 'milakat', 'jaidad',
            'dusra bhav', 'dhana bhav',
            'akrava bhav', 'labha bhav',
            'karz', 'runn', 'debt',
            'vaarsa', 'inheritance'
        ],
        'description': 'Wealth, finances, money, and property related matters',
        'always_include': False
    },
    
    'health': {
        'keywords': [
            # English - General
            'health', 'disease', 'illness', 'sickness', 'ailment',
            'medical', 'medicine', 'healthcare', 'treatment',
            'healing', 'cure', 'recovery', 'wellbeing', 'wellness',
            'fitness', 'vitality', 'stamina', 'energy',
            
            # English - Health conditions
            'chronic disease', 'acute disease', 'infection',
            'fever', 'cold', 'cough', 'flu', 'allergy',
            'diabetes', 'blood pressure', 'hypertension', 'heart disease',
            'cancer', 'tumor', 'kidney', 'liver', 'lung',
            'stomach', 'digestive', 'gastric', 'ulcer',
            'arthritis', 'joint pain', 'back pain', 'neck pain',
            'headache', 'migraine', 'paralysis', 'stroke',
            
            # English - Mental health
            'mental health', 'depression', 'anxiety', 'stress',
            'insanity', 'madness', 'psychological', 'psychiatric',
            'nervous disorder', 'mental disorder',
            
            # English - Medical events
            'surgery', 'operation', 'hospitalization', 'admission',
            'accident', 'injury', 'fracture', 'wound',
            'disability', 'handicap', 'impairment',
            
            # English - Longevity
            'longevity', 'lifespan', 'death', 'mortality',
            'life expectancy', 'premature death',
            
            # English - Houses
            '6th house', 'sixth house', 'house of disease',
            '8th house', 'eighth house', 'house of longevity',
            '1st house', 'first house', 'ascendant', 'physical body',
            
            # Sanskrit
            'roga', 'vyadhi', 'pida', 'dukha',
            'arogya', 'swasthya', 'niroga',
            'shashtam bhava', 'roga bhava', 'sixth house',
            'ashtam bhava', 'ayus bhava', 'eighth house',
            'tanu bhava', 'lagna', 'first house',
            'roga karaka', 'saturn', 'shani', 'mars', 'mangal',
            'ayur karaka', 'saturn', 'jupiter',
            'mritu', 'marana', 'death',
            'ayurveda', 'chikitsa', 'aushadhi',
            
            # Hindi
            'bimari', 'rog', 'beemari', 'dard', 'takleef',
            'swasthya', 'sehat', 'health', 'tandrusti',
            'dawaai', 'ilaj', 'upchar', 'medicine', 'treatment',
            'chhata ghar', '6th bhav', 'rog bhav',
            'aathva ghar', '8th bhav', 'mrityu bhav',
            'pahla ghar', 'lagna', 'ascendant',
            'lambi bimari', 'chronic disease',
            'dil ki bimari', 'heart disease',
            'sugar', 'madhumeh', 'diabetes',
            'blood pressure', 'bp', 'high bp',
            'cancer', 'kark rog',
            'surgery', 'operation', 'shastra kriya',
            'durghatna', 'accident', 'chot', 'injury',
            'maut', 'mrityu', 'death', 'marana',
            'aayushya', 'umar', 'lifespan', 'longevity',
            
            # Marathi
            'aajar', 'roga', 'aajarpan', 'ved', 'dukhane',
            'aarogya', 'prukti', 'tabbyet',
            'shastha bhav', 'roga bhav',
            'aathva bhav', 'aayu bhav',
            'upchar', 'daaru', 'aushadh',
            'accident', 'apaghat',
            'mrutyu', 'death', 'aayu'
        ],
        'description': 'Health, disease, medical matters, and longevity',
        'always_include': False
    },
    
    'education': {
        'keywords': [
            # English - General
            'education', 'study', 'studies', 'learning', 'knowledge',
            'student', 'pupil', 'scholar', 'academic', 'academics',
            'school', 'college', 'university', 'institute', 'academy',
            'degree', 'diploma', 'certificate', 'qualification',
            
            # English - Levels
            'primary education', 'secondary education', 'higher education',
            'graduation', 'undergraduate', 'post-graduation', 'postgraduate',
            'masters', 'phd', 'doctorate', 'research',
            
            # English - Exams
            'exam', 'examination', 'test', 'assessment',
            'competitive exam', 'entrance exam', 'board exam',
            'result', 'marks', 'grades', 'score', 'rank',
            'pass', 'fail', 'success', 'failure',
            
            # English - Subjects
            'science', 'mathematics', 'physics', 'chemistry', 'biology',
            'engineering', 'medical', 'commerce', 'arts', 'humanities',
            'literature', 'history', 'geography', 'economics',
            'computer science', 'it', 'technology',
            'law', 'management', 'mba',
            
            # English - Study aspects
            'concentration', 'focus', 'memory', 'intelligence',
            'brilliance', 'genius', 'wisdom', 'talent',
            'scholarship', 'merit', 'medal', 'topper',
            
            # English - Foreign education
            'foreign education', 'study abroad', 'overseas education',
            'international education', 'abroad study',
            
            # English - Houses
            '4th house', 'fourth house', 'house of learning',
            '5th house', 'fifth house', 'house of intelligence',
            '9th house', 'ninth house', 'higher learning',
            '2nd house', 'second house', 'early education',
            
            # Sanskrit
            'vidya', 'gyan', 'gyaan', 'pragya',
            'adhyayan', 'adhyayana', 'pathya', 'shiksha',
            'buddhi', 'medha', 'dhi', 'intelligence',
            'chaturthi bhava', 'vidya bhava', 'fourth house',
            'pancham bhava', 'buddhi bhava', 'fifth house',
            'navamsa bhava', 'dharma bhava', 'ninth house',
            'vidya karaka', 'jupiter', 'guru', 'mercury', 'budh',
            'saraswati', 'goddess of learning',
            
            # Hindi
            'padhai', 'study', 'shiksha', 'taleem', 'vidya',
            'gyan', 'gyaan', 'knowledge', 'padhna',
            'vidyarthi', 'student', 'chhatra',
            'school', 'college', 'vishwavidyalaya', 'university',
            'degree', 'upadi',
            'chautha ghar', '4th bhav', 'vidya bhav',
            'panchva ghar', '5th bhav', 'buddhi bhav',
            'nauva ghar', '9th bhav', 'dharma bhav',
            'exam', 'pariksha', 'imtihaan',
            'result', 'natija', 'parinaam',
            'pass', 'fail', 'safalta', 'asafalta',
            'honar', 'buddhi', 'buddhimaan', 'intelligent',
            'chhatrvritti', 'scholarship',
            'videsh me padhai', 'foreign education',
            
            # Marathi
            'shikshan', 'vidya', 'abhyas', 'gyan',
            'vidyarthi', 'vidyarthini', 'student',
            'shala', 'mahavidyalaya', 'vishwavidyalay',
            'chautha bhav', 'vidya bhav',
            'pachva bhav', 'buddhi bhav',
            'pariksha', 'exam', 'result',
            'buddhi', 'hushaar', 'intelligent',
            'shishyavrutti', 'scholarship'
        ],
        'description': 'Education, studies, learning, and academic matters',
        'always_include': False
    },
    
    'foreign': {
        'keywords': [
            # English - General
            'foreign', 'abroad', 'overseas', 'international',
            'foreign country', 'foreign land', 'foreign travel',
            'foreign settlement', 'foreign residence',
            
            # English - Travel
            'travel', 'journey', 'trip', 'tour', 'voyage',
            'long journey', 'short journey', 'pilgrimage',
            'world tour', 'international travel',
            
            # English - Settlement
            'settlement', 'migration', 'immigration', 'emigration',
            'relocation', 'moving abroad', 'settling abroad',
            'permanent residence', 'pr', 'citizenship',
            'green card', 'visa', 'work permit', 'work visa',
            
            # English - Purpose
            'work abroad', 'study abroad', 'foreign education',
            'foreign job', 'foreign employment',
            'business abroad', 'foreign business',
            
            # English - Locations
            'usa', 'america', 'uk', 'britain', 'canada', 'australia',
            'europe', 'asia', 'middle east', 'gulf',
            'dubai', 'uae', 'singapore',
            
            # English - Houses
            '12th house', 'twelfth house', 'house of foreign',
            '9th house', 'ninth house', 'long distance travel',
            '3rd house', 'third house', 'short travel',
            
            # Sanskrit
            'videsh', 'videsha', 'pardesh', 'pardesha',
            'pravasa', 'pravasa', 'yatra', 'tirtha yatra',
            'dwadash bhava', 'vyaya bhava', 'twelfth house',
            'navamsa bhava', 'dharma bhava', 'ninth house',
            'tritiya bhava', 'sahaj bhava', 'third house',
            'pravasa yoga', 'foreign yoga', 'videsh yoga',
            'rahu', 'ketu', 'foreign karakas',
            
            # Hindi
            'videsh', 'bahar desh', 'foreign', 'pardesh',
            'videsh yatra', 'foreign travel', 'travel',
            'videsh jana', 'going abroad',
            'videsh me rahna', 'living abroad', 'settle abroad',
            'videsh me naukri', 'foreign job', 'work abroad',
            'videsh me padhai', 'foreign education', 'study abroad',
            'barahva ghar', '12th bhav', 'vyaya bhav',
            'nauva ghar', '9th bhav', 'dharma bhav',
            'tisra ghar', '3rd bhav', 'sahaj bhav',
            'visa', 'green card', 'pr', 'citizenship',
            'america', 'canada', 'australia', 'dubai', 'singapore',
            
            # Marathi
            'videsh', 'videshi', 'pardes', 'paradesh',
            'videsh pravasa', 'pravas', 'yatra',
            'videsh settlement', 'sthalaantar',
            'barva bhav', 'vyaya bhav',
            'navva bhav', 'dharma bhav',
            'visa', 'pr', 'nagarikta'
        ],
        'description': 'Foreign travel, settlement abroad, and international matters',
        'always_include': False
    },
    
    'remedies': {
        'keywords': [
            # English - General
            'remedy', 'remedies', 'solution', 'solutions', 'fix',
            'cure', 'treatment', 'correction', 'mitigation',
            'relief', 'help', 'assistance',
            
            # English - Problem solving
            'how to fix', 'how to solve', 'how to improve', 'how to reduce',
            'how to strengthen', 'how to weaken', 'how to remove',
            'problem', 'issue', 'trouble', 'difficulty',
            'dosh', 'dosha', 'affliction', 'malefic effect',
            
            # Lal Kitab
            'lal kitab', 'laal kitab', 'red book', 'lalkitab',
            'totka', 'totke', 'quick remedy', 'simple remedy',
            
            # English - Remedy types
            'gemstone', 'ratna', 'stone', 'precious stone',
            'ruby', 'pearl', 'coral', 'emerald', 'yellow sapphire',
            'diamond', 'blue sapphire', 'hessonite', 'cats eye',
            
            # Donation and charity
            'donation', 'daan', 'charity', 'giving', 'alms',
            'food donation', 'clothes donation', 'money donation',
            
            # Worship and rituals
            'puja', 'pooja', 'worship', 'prayer', 'mantra',
            'japa', 'chanting', 'recitation',
            'homa', 'havan', 'yajna', 'yagya', 'fire ritual',
            'abhishek', 'abhishekam', 'aarti',
            
            # Yantra and symbols
            'yantra', 'yantram', 'mystical diagram',
            'talisman', 'amulet', 'kavach', 'tabeez',
            
            # Fasting
            'fasting', 'vrat', 'upvas', 'fast',
            'monday fast', 'tuesday fast', 'saturday fast',
            'ekadashi', 'pradosh', 'shivaratri',
            
            # Specific remedies
            'rudraksha', 'tulsi', 'basil', 'neem',
            'feeding', 'feed cow', 'feed birds', 'feed poor',
            'visit temple', 'temple visit', 'pilgrimage',
            
            # Planetary remedies
            'sun remedy', 'moon remedy', 'mars remedy', 'mercury remedy',
            'jupiter remedy', 'venus remedy', 'saturn remedy',
            'rahu remedy', 'ketu remedy',
            
            # Sanskrit
            'upay', 'upaya', 'parihara', 'parihaara',
            'prashanti', 'shanti', 'peace ritual',
            'mantra', 'stotra', 'stotram', 'hymn',
            'puja', 'archana', 'worship',
            'daan', 'dana', 'donation',
            'vrat', 'vrata', 'fast',
            'homa', 'yajna', 'fire ceremony',
            'ratna', 'mani', 'gemstone',
            'yantra', 'mystical diagram',
            'tantra', 'esoteric practice',
            'japa', 'chanting', 'mantra japa',
            
            # Doshas to fix
            'mangal dosh', 'manglik dosh', 'kaal sarp dosh',
            'pitra dosh', 'grahan dosh', 'nadi dosh',
            'shani sade sati', 'sade sati', 'shani dhaiya',
            'rahu kaal', 'rahu kalam',
            
            # Hindi
            'upay', 'upaye', 'totka', 'totke', 'ilaj',
            'hal', 'solution', 'samadhan',
            'dosh', 'dosh nivaran', 'dosh ka upay',
            'ratna', 'mani', 'gemstone', 'patthar',
            'manik', 'ruby', 'moti', 'pearl', 'moonga', 'coral',
            'panna', 'emerald', 'pukhraj', 'yellow sapphire',
            'heera', 'diamond', 'neelam', 'blue sapphire',
            'daan', 'dan', 'donation', 'charity', 'daan-punya',
            'puja', 'pooja', 'aaradhana', 'worship',
            'mantra', 'jaap', 'jap', 'chanting',
            'havan', 'homa', 'yagy', 'yagya',
            'yantra', 'kavach', 'taweez',
            'vrat', 'upvas', 'fast', 'roza',
            'somvaar ka vrat', 'shanivar ka vrat',
            'rudraksh', 'rudraksha',
            'mandir jana', 'temple visit',
            'gau seva', 'cow service', 'gaay ko khana',
            'bhookhe ko khana', 'feed poor',
            
            # Marathi
            'upay', 'totaka', 'totake',
            'dosh', 'dosh nivaran',
            'ratna', 'mani', 'ratan',
            'daan', 'dan', 'datvya',
            'puja', 'pooja', 'archana',
            'mantra', 'jap', 'jaap',
            'homa', 'havan', 'yadnya',
            'yantra', 'kavach',
            'upvas', 'vrata', 'fast',
            'rudraksh', 'tulsi',
            'mandir', 'temple', 'devasthan'
        ],
        'description': 'Astrological remedies, solutions, and corrective measures',
        'always_include': False
    },
    
    'timing': {
        'keywords': [
            # English - General
            'timing', 'time', 'when', 'period', 'duration',
            'date', 'year', 'month', 'age',
            'prediction', 'forecast', 'future',
            
            # English - Dasha system
            'dasha', 'mahadasha', 'antardasha', 'pratyantar dasha',
            'planetary period', 'sub-period', 'major period',
            'sun dasha', 'moon dasha', 'mars dasha', 'rahu dasha',
            'jupiter dasha', 'saturn dasha', 'mercury dasha',
            'venus dasha', 'ketu dasha',
            
            # English - Transits
            'transit', 'transits', 'gochara', 'gochar',
            'saturn transit', 'jupiter transit', 'rahu transit',
            'planetary transit', 'current transit',
            
            # English - Timing questions
            'when will', 'when to', 'what time', 'which year',
            'which age', 'how long', 'how soon',
            'best time', 'auspicious time', 'good time',
            'muhurat', 'muhurta', 'auspicious moment',
            
            # English - Specific timings
            'marriage timing', 'job timing', 'child birth timing',
            'property purchase timing', 'vehicle purchase timing',
            'travel timing', 'surgery timing',
            
            # Sanskrit
            'dasha', 'mahadasha', 'antardasha',
            'vimshottari dasha', 'ashtottari dasha',
            'gochara', 'gocharaphala', 'transit',
            'muhurta', 'muhurat', 'auspicious time',
            'kaal', 'samaya', 'time', 'period',
            'bhukti', 'sub-period',
            
            # Hindi
            'samay', 'waqt', 'time', 'timing',
            'kab', 'when', 'kab hoga', 'when will happen',
            'kis umar me', 'kis saal me', 'which age', 'which year',
            'dasha', 'mahadasha', 'antardasha',
            'surya dasha', 'chandra dasha', 'mangal dasha',
            'budh dasha', 'guru dasha', 'shukra dasha',
            'shani dasha', 'rahu dasha', 'ketu dasha',
            'gochar', 'gochara', 'transit',
            'shani ki sade sati', 'shani ki dhaiya',
            'guru ki dasha', 'shani ki dasha',
            'shubh samay', 'auspicious time', 'achha waqt',
            'muhurat', 'shubh muhurat',
            
            # Marathi
            'samay', 'vel', 'time', 'timing',
            'kevha', 'when', 'koni veles',
            'dasha', 'mahadasha', 'antardasha',
            'gochar', 'transit',
            'shubh muhurt', 'muhurt'
        ],
        'description': 'Timing predictions, dasha periods, and transit effects',
        'always_include': False
    },
    
    'compatibility': {
        'keywords': [
            # English - General
            'compatibility', 'matching', 'match', 'kundali matching',
            'horoscope matching', 'chart matching',
            'partner compatibility', 'couple compatibility',
            
            # English - Relationship compatibility
            'love compatibility', 'marriage compatibility',
            'sexual compatibility', 'emotional compatibility',
            'mental compatibility', 'intellectual compatibility',
            'financial compatibility',
            
            # English - Matching systems
            'guna milan', 'ashtakoot', 'ashta koota',
            'points', 'score', 'matching points',
            '36 points', '36 gunas',
            
            # English - Aspects checked
            'varna', 'vashya', 'tara', 'yoni', 'graha maitri',
            'gana', 'bhakoot', 'nadi',
            'manglik', 'mangal dosh', 'kuja dosha',
            
            # English - Business/Partnership
            'business partnership', 'business partner',
            'partnership compatibility', 'partner matching',
            
            # Sanskrit
            'kundali milan', 'kundali matching',
            'guna milan', 'guna milana',
            'ashtakoot', 'ashta kuta', 'eight compatibilities',
            'varna koot', 'vashya koot', 'tara koot',
            'yoni koot', 'graha maitri koot',
            'gana koot', 'bhakoot', 'nadi koot',
            'mangal dosha', 'kuja dosha', 'manglik',
            
            # Hindi
            'mil', 'milan', 'matching', 'mel',
            'kundali milan', 'kundali matching', 'guna milan',
            'gun milan', '36 gun', '36 gunas',
            'jodi', 'pair', 'couple',
            'manglik', 'mangal dosh', 'non-manglik',
            'varna', 'vashya', 'tara', 'yoni',
            'gana', 'bhakoot', 'nadi',
            'acchi jodi', 'buri jodi', 'good match', 'bad match',
            
            # Marathi
            'jatak milan', 'kundali milan',
            'guna milan', 'gun milan',
            'jodi', 'matching',
            'manglik', 'mangal dosh'
        ],
        'description': 'Compatibility matching for marriage and partnerships',
        'always_include': False
    },
    
    'muhurat': {
        'keywords': [
            # English - General
            'muhurat', 'muhurta', 'auspicious time', 'auspicious date',
            'good time', 'best time', 'favorable time',
            'election', 'electional astrology',
            
            # English - Event types
            'marriage muhurat', 'wedding muhurat',
            'griha pravesh', 'house warming', 'housewarming',
            'vehicle purchase', 'car purchase', 'bike purchase',
            'property purchase', 'home purchase', 'house purchase',
            'business opening', 'shop opening', 'office opening',
            'name ceremony', 'naming ceremony', 'naamkaran',
            'thread ceremony', 'upanayana', 'sacred thread',
            'mundan', 'tonsure', 'first haircut',
            'surgery date', 'operation date',
            'travel date', 'journey date', 'trip date',
            'exam date', 'interview date', 'job joining date',
            
            # English - Specific occasions
            'engagement date', 'betrothal date',
            'conception time', 'child planning',
            'inauguration', 'foundation laying',
            'signing contract', 'deal closing',
            
            # Sanskrit
            'muhurta', 'shubh muhurta', 'auspicious moment',
            'vivaha muhurta', 'marriage muhurat',
            'griha pravesh', 'gruha pravesha', 'house entry',
            'vastu shanti', 'peace ceremony',
            'naamkaran', 'naming ceremony',
            'upanayana', 'sacred thread ceremony',
            'mundan', 'chudakarana', 'tonsure ceremony',
            'annaprashan', 'first feeding ceremony',
            
            # Hindi
            'shubh muhurat', 'achha samay', 'auspicious time',
            'shaadi ka muhurat', 'vivah muhurat',
            'griha pravesh', 'gruh pravesh', 'ghar me pravesh',
            'gaadi khareedna', 'car purchase', 'vehicle',
            'ghar khareedna', 'jaidad khareedna', 'property',
            'dukaan kholna', 'vyapar shuru', 'business opening',
            'naamkaran', 'naam rakhna', 'naming',
            'mundan', 'chhoti', 'hair cutting',
            'annaprashan', 'pahla khaana',
            'yatra', 'safar', 'journey', 'travel',
            
            # Marathi
            'shubh muhurt', 'chaan vel',
            'lagna muhurt', 'vivah muhurt',
            'gruha pravesh', 'ghar pravesh',
            'vaahan kharidne', 'vehicle purchase',
            'dukaan ughadan', 'vyavasay shuruvatil',
            'naav thevane', 'naming',
            'mundan', 'kesavasran',
            'pravas', 'journey'
        ],
        'description': 'Auspicious timings and electional astrology for events',
        'always_include': False
    },
    
    'spirituality': {
        'keywords': [
            # English - General
            'spirituality', 'spiritual', 'moksha', 'liberation',
            'enlightenment', 'salvation', 'nirvana',
            'meditation', 'yoga', 'sadhana', 'spiritual practice',
            'devotion', 'bhakti', 'worship', 'prayer',
            
            # English - Religious aspects
            'religion', 'religious', 'faith', 'belief',
            'dharma', 'righteousness', 'duty',
            'karma', 'past life', 'rebirth', 'reincarnation',
            'destiny', 'fate', 'prarabdha',
            
            # English - Spiritual progress
            'spiritual growth', 'spiritual journey',
            'self-realization', 'god realization',
            'guru', 'teacher', 'master', 'mentor',
            'pilgrimage', 'holy place', 'sacred place',
            
            # English - Occult
            'psychic', 'intuition', 'third eye',
            'occult', 'esoteric', 'mystical',
            'astral', 'supernatural', 'paranormal',
            
            # English - Houses
            '9th house', 'ninth house', 'dharma house',
            '12th house', 'twelfth house', 'moksha house',
            '5th house', 'fifth house', 'poorva punya',
            
            # Sanskrit
            'adhyatma', 'adhyatmik', 'spirituality',
            'moksha', 'mukti', 'kaivalya', 'liberation',
            'dharma', 'righteousness', 'duty',
            'karma', 'karmic', 'prarabdha',
            'punarjanma', 'rebirth', 'reincarnation',
            'tapasya', 'penance', 'austerity',
            'sadhana', 'spiritual practice',
            'bhakti', 'devotion', 'prema',
            'jnana', 'gyan', 'knowledge', 'wisdom',
            'yoga', 'meditation', 'dhyana',
            'guru', 'spiritual teacher', 'acharya',
            'tirtha', 'pilgrimage', 'holy place',
            'navamsa bhava', 'dharma bhava', 'ninth house',
            'dwadash bhava', 'moksha bhava', 'twelfth house',
            'pancham bhava', 'poorva punya bhava', 'fifth house',
            'jupiter', 'guru', 'spiritual karaka',
            'ketu', 'moksha karaka',
            
            # Hindi
            'adhyatmik', 'adhyatma', 'spirituality',
            'moksh', 'mukti', 'liberation', 'nirvana',
            'dharm', 'dharma', 'duty', 'righteousness',
            'karma', 'punarjanm', 'rebirth',
            'bhakti', 'devotion', 'worship', 'puja',
            'gyan', 'gyaan', 'knowledge', 'wisdom',
            'yog', 'yoga', 'meditation', 'dhyan',
            'sadhna', 'sadhana', 'tapasya', 'tap',
            'guru', 'spiritual guru', 'guruji',
            'tirtha', 'teerth', 'pilgrimage',
            'nauva ghar', '9th bhav', 'dharma bhav',
            'barahva ghar', '12th bhav', 'moksha bhav',
            'panchva ghar', '5th bhav', 'poorva punya',
            
            # Marathi
            'adhyatmik', 'adhyatma',
            'moksha', 'mukti',
            'dharma', 'dharmik',
            'karma', 'punarjanma',
            'bhakti', 'upasana',
            'dnyan', 'gyan',
            'yog', 'dhyan',
            'sadhana', 'tapasya',
            'guru', 'guruji',
            'tirtha', 'pilgrimage'
        ],
        'description': 'Spirituality, moksha, dharma, and religious matters',
        'always_include': False
    }
}


# Helper functions

def get_all_domains() -> list:
    """Get list of all domain names"""
    return list(DOMAIN_DEFINITIONS.keys())


def get_domain_info(domain: str) -> dict:
    """Get information about a specific domain"""
    return DOMAIN_DEFINITIONS.get(domain, {})


def get_domain_keywords(domain: str) -> list:
    """Get keywords for a specific domain"""
    return DOMAIN_DEFINITIONS.get(domain, {}).get('keywords', [])


def get_always_include_domains() -> list:
    """Get list of domains that should always be included"""
    return [
        domain for domain, info in DOMAIN_DEFINITIONS.items()
        if info.get('always_include', False)
    ]


def search_keyword_in_domains(keyword: str) -> list:
    """Find which domains contain a specific keyword"""
    keyword_lower = keyword.lower()
    matching_domains = []
    
    for domain, info in DOMAIN_DEFINITIONS.items():
        if any(keyword_lower in kw.lower() for kw in info['keywords']):
            matching_domains.append(domain)
    
    return matching_domains


# Print statistics when module is imported
if __name__ == "__main__":
    print("="*60)
    print("DOMAIN DEFINITIONS - Statistics")
    print("="*60)
    
    for domain, info in DOMAIN_DEFINITIONS.items():
        keyword_count = len(info['keywords'])
        always_include = "✓" if info.get('always_include', False) else " "
        
        print(f"\n[{always_include}] {domain.upper()}")
        print(f"    Description: {info['description']}")
        print(f"    Keywords: {keyword_count}")
        print(f"    Sample keywords: {', '.join(info['keywords'][:5])}...")
    
    print(f"\n{'='*60}")
    print(f"Total Domains: {len(DOMAIN_DEFINITIONS)}")
    print(f"Always Include: {len(get_always_include_domains())}")
    print(f"Total Keywords: {sum(len(info['keywords']) for info in DOMAIN_DEFINITIONS.values())}")
    print(f"{'='*60}")