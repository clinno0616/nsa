import numpy as np
from faker import Faker
import random
import json

class ChineseDataGenerator:
    def __init__(self):
        self.fake = Faker(['zh_TW'])

        # 科學領域詞彙
        self.science_terms = {
            '天文': [
                "星系", "黑洞", "量子糾纏", "暗物質", "中子星", "類星體",
                "星雲", "銀河系", "太陽風", "引力波", "宇宙射線", "星際物質",
                "超新星", "白矮星", "紅巨星", "光年", "奇點", "空間站"
            ],
            '物理': [
                "量子力學", "相對論", "粒子物理", "波函數", "反物質",
                "超導體", "電磁場", "核裂變", "核融合", "質能轉換",
                "光電效應", "波粒二象性", "熱力學", "重力場", "動量守恆"
            ],
            '化學': [
                "化學鍵", "催化劑", "氧化還原", "分子結構", "化學平衡",
                "有機化合物", "無機化合物", "酸鹼反應", "電解質", "化學動力學",
                "原子軌道", "共價鍵", "離子鍵", "氫鍵", "范德華力"
            ],
            '占星': [
                "太陽宮位", "月亮相位", "上升星座", "命盤解讀", "星盤排列",
                "黃道十二宮", "行星相位", "本命盤", "宮位主宰", "行運圖",
                "四元素", "三大守護", "月相週期", "星座特質", "行星能量"
            ],
            '數學': [
                "微積分", "線性代數", "概率論", "拓撲學", "數論",
                "複變函數", "泛函分析", "群論", "範疇論", "邏輯學",
                "矩陣運算", "向量空間", "傅立葉變換", "微分方程", "最優化"
            ],
            '電子': [
                "電路設計", "數位邏輯", "類比電路", "微處理器", "半導體",
                "電容器", "電感器", "電阻器", "二極管", "運算放大器",
                "電壓調節", "頻率響應", "濾波電路", "振盪器", "數位信號"
            ],
            '資訊': [
                "演算法", "資料結構", "機器學習", "人工智慧", "雲計算",
                "區塊鏈", "網路安全", "大數據", "軟體工程", "作業系統",
                "編譯原理", "資料庫", "網路架構", "系統設計", "程式語言"
            ],
            '統計': [
                "統計推論", "假設檢定", "迴歸分析", "時間序列", "抽樣理論",
                "變異數分析", "多變量分析", "貝氏統計", "非參數統計", "實驗設計",
                "因素分析", "主成分分析", "聚類分析", "存活分析", "品質管制"
            ],
            '經濟': [
                "供需理論", "市場均衡", "宏觀經濟", "微觀經濟", "貨幣政策",
                "財政政策", "國際貿易", "產業經濟", "金融市場", "投資理論",
                "經濟成長", "通貨膨脹", "匯率機制", "經濟週期", "資本市場"
            ]
        }

        # 科學相關模板
        self.science_templates = {
            '天文': [
                "在{astronomical_object}的研究中，科學家發現{discovery}，這暗示著{implication}。",
                "通過觀測{celestial_event}，天文學家們提出了關於{theory}的新見解。",
                "最新的{space_research}表明，{finding}可能改變我們對宇宙的認知。"
            ],
            '物理': [
                "根據{physics_theory}，當{condition}時，會發生{phenomenon}。",
                "在{experiment}實驗中，研究人員觀察到{observation}，這證實了{conclusion}。",
                "{physical_law}告訴我們，{principle}與{result}之間存在密切關係。"
            ],
            '化學': [
                "研究發現，{chemical_compound}在{reaction_condition}下，會產生{reaction_result}。",
                "通過分析{molecular_structure}，科學家理解了{chemical_property}的本質。",
                "在{chemical_process}過程中，{catalyst}的作用機理揭示了{mechanism}。"
            ],
            '數學': [
                "利用{math_theory}，我們可以證明{theorem}成立的條件是{condition}。",
                "在{mathematical_field}中，{principle}的應用可以解決{problem}。",
                "通過{mathematical_method}，可以得出{conclusion}的重要結論。"
            ]
        }

        # 擴展模板類型
        self.template_groups = {
            '科學': [
                "在{field}領域中，{theory}的發展揭示了{discovery}的奧秘。",
                "根據{research}的研究，{phenomenon}的形成與{factor}密切相關。",
                "通過{method}分析，科學家們發現{finding}具有{property}的特性。",
                "在{experiment}實驗中，{observation}表明{conclusion}的正確性。",
                "從{perspective}的角度來看，{subject}與{correlation}存在重要關聯。"
            ],
            '技術': [
                "最新的{technology}技術突破，為{application}帶來了{impact}。",
                "在{field}領域，{innovation}的應用實現了{achievement}。",
                "通過{method}方法，{problem}得到了{solution}的解決方案。"
            ],
            '分析': [
                "根據{data}分析，{trend}顯示出{pattern}的特徵。",
                "從{perspective}來看，{phenomenon}反映了{insight}的本質。",
                "透過{method}研究，{subject}展現出{characteristic}的規律。"
            ],
            '理論': [
                "在{theory}中，{concept}的定義基於{foundation}。",
                "{principle}說明了{phenomenon}與{factor}的關係。",
                "根據{law}，{condition}會導致{result}的出現。"
            ],
            '應用': [
                "將{theory}應用於{field}，可以解決{problem}。",
                "在{industry}中，{technology}的實施改善了{aspect}。",
                "通過{method}的應用，{task}達到了{goal}的效果。"
            ],
            '描述': [
                "這是一個{adj}的{time}，{scene}，讓人感到{feeling}。",
                "在{location}的{time}，{scene}，給人一種{feeling}的感覺。",
                "{time}的{location}，{scene}，{result}。"
            ],
            '對話': [
                "{person_a}對{person_b}說：「{dialogue}」，{reaction}。",
                "「{dialogue}」{person_a}溫和地對{person_b}說道，{reaction}。",
                "{person_a}和{person_b}討論著{topic}，{dialogue_scene}。"
            ],
            '議論': [
                "關於{topic}，我認為{opinion}，因為{reason}。",
                "{topic}這個問題，需要從{aspect}來看，{analysis}。",
                "探討{topic}時，應該注意{point}，這樣才能{goal}。"
            ],
            '記敘': [
                "記得那年{time}，{event}，讓我{feeling}。",
                "經歷了{event}後，我深深體會到{insight}。",
                "每當想起{event}，總會{reaction}，尤其是{detail}。"
            ],
            '抒情': [
                "面對{situation}，內心充滿了{emotion}，不禁想到{thought}。",
                "{scene}的景象，觸動了我的{emotion}，讓我想起{memory}。",
                "在{situation}的時候，常常會{action}，感受著{feeling}。"
            ]
        }

        # 擴展各類詞彙庫
        self.words = {
            'time': [
                "清晨", "黃昏", "深夜", "正午", "傍晚", "凌晨", 
                "春天", "夏天", "秋天", "冬天",
                "週末", "假期", "節日", "工作日", "課餘時光"
            ],
            'location': [
                "公園", "校園", "咖啡廳", "圖書館", "海邊", "山上",
                "街角", "廣場", "小巷", "天臺", "庭院", "市集",
                "教室", "辦公室", "車站", "機場", "商場"
            ],
            'scene': [
                "陽光灑落在石板路上", "微風輕拂過樹梢", "雨滴敲打著窗戶",
                "人們匆匆走過", "落葉輕輕飄落", "花朵綻放著笑臉",
                "街燈漸漸亮起", "鳥兒在枝頭歌唱", "雲彩變幻著形狀"
            ],
            'feeling': [
                "溫馨", "愉快", "感動", "平靜", "舒適", "幸福",
                "充實", "期待", "懷念", "感激", "欣慰", "寧靜",
                "雀躍", "溫暖", "愜意", "滿足", "喜悅"
            ],
            'topic': [
                "教育改革", "環境保護", "科技創新", "文化傳承", "社會發展",
                "健康生活", "職業規劃", "人際關係", "城市發展", "藝術創作",
                "經濟趨勢", "生活方式", "學習方法", "工作態度"
            ],
            'person_type': [
                "老師", "同學", "朋友", "長輩", "同事", "鄰居",
                "陌生人", "專家", "前輩", "後輩", "親戚", "客人"
            ],
            'emotion': [
                "喜悅", "感動", "期待", "懷念", "感激", "欣慰",
                "好奇", "興奮", "平靜", "滿足", "溫暖", "敬佩"
            ]
        }

    def generate_scientific_content(self, field):
        """生成特定科學領域的內容"""
        if field not in self.science_terms:
            field = random.choice(list(self.science_terms.keys()))
            
        terms = self.science_terms[field]
        
        content = {
            'field': field,
            'theory': random.choice(terms),
            'discovery': f"關於{random.choice(terms)}的新發現",
            'phenomenon': random.choice(terms),
            'factor': random.choice(terms),
            'method': random.choice([
                "實驗觀察", "數據分析", "理論推導", "模型模擬", 
                "系統研究", "對比研究", "跨學科研究"
            ]),
            'finding': f"{random.choice(terms)}的特性",
            'property': random.choice([
                "普遍性", "特殊性", "關聯性", "可預測性", 
                "穩定性", "變異性", "週期性"
            ]),
            'perspective': random.choice([
                "理論", "實踐", "應用", "發展", "創新",
                "系統", "整體", "微觀", "宏觀"
            ]),
            'subject': random.choice(terms),
            'correlation': random.choice([
                "因果關係", "相關性", "影響機制", "互動規律",
                "發展趨勢", "變化規律", "作用機理"
            ]),
            'conclusion': random.choice([
                "研究假設", "理論預測", "實驗結果", "分析結論",
                "觀察發現", "推理結果", "驗證結果"
            ])
        }
        
        return content
    def generate_technical_content(self):
        """生成技術相關內容"""
        fields = list(self.science_terms.keys())
        field = random.choice(fields)
        terms = self.science_terms[field]
        
        return {
            'technology': random.choice(terms),
            'application': random.choice([
                "產業升級", "技術創新", "效率提升", "成本降低",
                "品質改善", "流程優化", "服務提升"
            ]),
            'impact': random.choice([
                "革命性的變革", "顯著的進步", "重大的突破",
                "全新的可能", "深遠的影響", "巨大的潛力"
            ]),
            'field': field,
            'innovation': random.choice(terms),
            'achievement': random.choice([
                "效率的提升", "品質的改善", "成本的降低",
                "性能的優化", "功能的擴展", "使用的便利"
            ]),
            'method': random.choice([
                "系統化", "智能化", "自動化", "數位化",
                "網路化", "模組化", "整合化"
            ]),
            'problem': random.choice([
                "技術瓶頸", "效率問題", "質量控制", "成本管理",
                "資源配置", "系統整合", "過程優化"
            ]),
            'solution': random.choice([
                "創新方案", "優化方法", "改進措施", "解決途徑",
                "突破方向", "應對策略", "處理方式"
            ])
        }


    def generate_topic(self):
        topics = ["教育", "科技", "環境", "文化", "健康", "經濟", "社會", "藝術", "創新", "傳統", 
                 "人工智能", "可持續發展", "心理健康", "數字化轉型", "文化傳承", "社會公平"]
        points = ["持續學習", "與時俱進", "永續發展", "傳承創新", "平衡發展", "共同參與",
                 "系統思考", "跨界合作", "創新突破", "深度理解", "實踐驗證", "價值創造"]
        return {
            "topic": random.choice(topics),
            "point": random.choice(points)
        }

    def generate_quote(self):
        topics = ["教育", "科技", "環境", "文化", "健康", "經濟", "社會", "藝術", "創新", "傳統",
                 "領導力", "創造力", "學習能力", "適應力", "溝通能力", "思考能力"]
        people = ["古人", "專家", "學者", "先賢", "老師", "前輩", "哲人", "思想家", "創新者", "實踐者"]
        quotes = [
            "知識就像一座寶山，需要我們不斷探索",
            "創新是推動發展的核心動力",
            "文化是民族的靈魂所在",
            "教育是改變命運的重要力量",
            "健康是人生最寶貴的財富",
            "智慧來自於持續的學習與思考",
            "成功是堅持的結果",
            "生活中的每個細節都值得關注",
            "真理越辯越明",
            "實踐是檢驗真理的唯一標準"
        ]
        return {
            "topic": random.choice(topics),
            "person": random.choice(people),
            "quote": random.choice(quotes)
        }

    def generate_learning(self):
        subjects = ["程式設計", "外語", "音樂", "繪畫", "寫作", "攝影", "烹飪", "瑜伽", "投資", "心理學"]
        discoveries = [
            "學習方法比內容更重要",
            "熱情是最好的老師",
            "失敗是成功的墊腳石",
            "堅持就是勝利",
            "知識需要在實踐中驗證"
        ]
        feelings = ["深受啟發", "充滿動力", "茅塞頓開", "信心倍增", "重新認識自己"]
        return {
            "subject": random.choice(subjects),
            "discovery": random.choice(discoveries),
            "learning_feeling": random.choice(feelings)
        }

    def generate_relationship(self):
        relationships = ["家人", "朋友", "同事", "老師", "鄰居", "同學"]
        activities = ["聊天", "吃飯", "旅行", "運動", "學習", "工作"]
        interactions = [
            "分享生活點滴",
            "互相鼓勵",
            "交流心得",
            "增進感情",
            "建立信任"
        ]
        return {
            "relationship": random.choice(relationships),
            "activity": random.choice(activities),
            "interaction": random.choice(interactions)
        }

    def generate_society(self):
        phenomena = ["社交媒體", "遠程工作", "終身學習", "共享經濟", "數字化生活", "環保意識"]
        insights = [
            "人際關係的轉變",
            "工作方式的革新",
            "生活理念的更新",
            "價值觀的多元化",
            "社會結構的變遷"
        ]
        return {
            "social_phenomenon": random.choice(phenomena),
            "social_insight": random.choice(insights)
        }

    def generate_emotion(self):
        triggers = ["意外驚喜", "久別重逢", "目標達成", "收到感謝", "幫助他人", "克服困難"]
        responses = [
            "感動不已",
            "欣喜若狂",
            "心潮澎湃",
            "溫暖幸福",
            "充滿希望"
        ]
        return {
            "emotion_trigger": random.choice(triggers),
            "emotional_response": random.choice(responses)
        }

    def generate_dream(self):
        dream_types = ["職業", "生活", "創業", "公益", "藝術", "教育"]
        contents = [
            "創立自己的公司",
            "環遊世界",
            "出版一本書",
            "幫助更多人",
            "成為專業音樂家"
        ]
        dedications = [
            "付出所有努力",
            "堅持不懈",
            "克服一切困難",
            "持續學習進步",
            "永不放棄"
        ]
        return {
            "dream_type": random.choice(dream_types),
            "dream_content": random.choice(contents),
            "dedication": random.choice(dedications)
        }

    def generate_experience(self):
        experiences = ["實習", "志願服務", "創業", "比賽", "留學", "工作"]
        lessons = [
            "團隊合作的重要性",
            "溝通技巧的提升",
            "時間管理的藝術",
            "堅持的價值",
            "創新思維的重要性"
        ]
        return {
            "experience": random.choice(experiences),
            "lesson": random.choice(lessons)
        }

    def generate_friendship(self):
        friend_types = ["知心", "童年", "學習", "工作", "網路", "運動"]
        actions = ["分享心事", "一起奮鬥", "互相支持", "共同成長", "切磋交流"]
        impacts = [
            "讓人感到溫暖",
            "帶來新的視角",
            "增添生活樂趣",
            "激發潛能",
            "建立深厚友誼"
        ]
        return {
            "friend_type": random.choice(friend_types),
            "action": random.choice(actions),
            "impact": random.choice(impacts)
        }

    def generate_daily_inspiration(self):
        daily_things = ["早晨陽光", "路人微笑", "孩童笑聲", "街角咖啡", "書本故事", "音樂旋律"]
        inspirations = [
            "生活的美好",
            "希望的力量",
            "簡單的幸福",
            "前進的動力",
            "創造的靈感"
        ]
        return {
            "daily_thing": random.choice(daily_things),
            "inspiration": random.choice(inspirations)
        }

    def generate_challenge(self):
        challenges = ["工作壓力", "學習困境", "人際關係", "生活變故", "健康問題"]
        solutions = [
            "積極面對",
            "尋求幫助",
            "調整心態",
            "制定計劃",
            "堅持不懈"
        ]
        reasons = [
            "這樣最有效率",
            "能學到更多",
            "可以持續發展",
            "對大家都好",
            "符合長期目標"
        ]
        return {
            "challenge": random.choice(challenges),
            "solution": random.choice(solutions),
            "reason": random.choice(reasons)
        }

    def generate_festival(self):
        festivals = ["春節", "中秋節", "端午節", "元宵節", "清明節"]
        customs = ["團圓飯", "賞月", "包粽子", "燈謎", "掃墓"]
        feelings = [
            "溫馨幸福",
            "文化自豪",
            "思念情深",
            "傳統情懷",
            "團圓喜悅"
        ]
        return {
            "festival": random.choice(festivals),
            "custom": random.choice(customs),
            "cultural_feeling": random.choice(feelings)
        }

    def generate_traditional_art(self):
        art_forms = ["書法", "國畫", "戲曲", "園林", "建築", "音樂"]
        values = [
            "天人合一",
            "以和為貴",
            "寫意精神",
            "匠心獨運",
            "生生不息"
        ]
        return {
            "art_form": random.choice(art_forms),
            "cultural_value": random.choice(values)
        }

    def generate_nature(self):
        elements = ["四季更替", "日月運行", "花開花落", "潮起潮落", "雲卷雲舒"]
        wisdoms = [
            "生命的循環",
            "時間的力量",
            "自然的規律",
            "萬物的和諧",
            "生態的平衡"
        ]
        return {
            "nature_element": random.choice(elements),
            "nature_wisdom": random.choice(wisdoms)
        }

    def generate_environment(self):
        aspects = ["空氣", "水源", "森林", "海洋", "土壤"]
        actions = [
            "減少浪費",
            "循環利用",
            "節約能源",
            "生態保護",
            "環保教育"
        ]
        results = [
            "永續發展",
            "改善環境",
            "保護生態",
            "減少污染",
            "維護地球"
        ]
        return {
            "environment_aspect": random.choice(aspects),
            "action": random.choice(actions),
            "result": random.choice(results)
        }

    def generate_reflection(self):
        events = ["求學經歷", "工作歷程", "人生選擇", "重要決定", "成長過程"]
        reflections = [
            "更加成熟",
            "深感欣慰",
            "充滿感激",
            "獲益良多",
            "豁然開朗"
        ]
        return {
            "past_event": random.choice(events),
            "reflection": random.choice(reflections)
        }

    def generate_time(self):
        metaphors = ["流水", "白駒過隙", "光影", "沙漏", "長河"]
        feelings = [
            "感慨萬千",
            "珍惜當下",
            "憧憬未來",
            "回味過往",
            "領悟人生"
        ]
        return {
            "metaphor": random.choice(metaphors),
            "time_feeling": random.choice(feelings)
        }

    def generate_trend(self):
        fields = ["教育", "科技", "文化", "經濟", "社會"]
        trends = [
            "數字化轉型",
            "可持續發展",
            "創新驅動",
            "共享經濟",
            "智能化升級"
        ]
        indications = [
            "發展方向的轉變",
            "社會進步的必然",
            "時代的要求",
            "未來的機遇",
            "變革的動力"
        ]
        return {
            "field": random.choice(fields),
            "trend": random.choice(trends),
            "indication": random.choice(indications)
        }

    def generate_mood(self):
        moods = ["開心", "感慨", "思念", "期待", "平靜"]
        desires = [
            "與朋友分享",
            "安靜思考",
            "寫些文字",
            "放鬆心情",
            "計劃未來"
        ]
        return {
            "mood": random.choice(moods),
            "desire": random.choice(desires)
        }

    def generate_reading(self):
        book_types = ["文學", "哲學", "科普", "歷史", "藝術"]
        effects = [
            "獲得新知識",
            "改變思維方式",
            "增長見識",
            "啟發靈感",
            "放鬆心情"
        ]
        return {
            "book_type": random.choice(book_types),
            "reading_effect": random.choice(effects)
        }

    def generate_literature(self):
        authors = ["魯迅", "張愛玲", "余華", "莫言", "王安憶"]
        feelings = [
            "深受感動",
            "思維擴展",
            "靈魂觸動",
            "審美提升",
            "人生感悟"
        ]
        points = [
            "對人性的洞察",
            "文字的魅力",
            "故事的張力",
            "情節的安排",
            "主題的深度"
        ]
        return {
            "author": random.choice(authors),
            "literary_feeling": random.choice(feelings),
            "specific_point": random.choice(points)
        }

    def generate_scene(self):
        locations = ["城市", "鄉村", "校園", "公園", "海邊", "山上"]
        actions = ["散步", "閱讀", "思考", "交談", "觀察", "學習"]
        thoughts = [
            "生活的真諦",
            "人生的意義",
            "社會的發展",
            "文化的傳承",
            "自然的美好"
        ]
        return {
            "location": random.choice(locations),
            "action": random.choice(actions),
            "thought": random.choice(thoughts)
        }
        
    def generate_season_scene(self):
        seasons = ["春天", "夏天", "秋天", "冬天"]
        scenes = [
            "花朵綻放，生機盎然",
            "陽光明媚，萬物生長",
            "落葉紛飛，景色怡人",
            "白雪皚皚，寧靜祥和"
        ]
        feelings = ["溫暖", "愉快", "感動", "平靜", "舒適", "美好"]
        return {
            "season": random.choice(seasons),
            "scene": random.choice(scenes),
            "feeling": random.choice(feelings)
        }
        
    def generate_memory(self):
        memories = ["童年往事", "學習經歷", "工作體驗", "生活點滴", "旅行見聞"]
        emotions = ["感慨萬分", "心潮澎湃", "回味無窮", "倍感珍惜", "深受啟發"]
        situations = ["獨處時", "忙碌時", "放鬆時", "思考時", "交流時"]
        return {
            "memory": random.choice(memories),
            "emotion": random.choice(emotions),
            "situation": random.choice(situations)
        }

    def generate_goal(self):
        goals = [
            "職業發展",
            "個人成長",
            "健康管理",
            "財務規劃",
            "技能提升",
            "人際關係",
            "家庭幸福",
            "社會貢獻"
        ]
        key_points = [
            "持之以恆",
            "循序漸進",
            "不斷學習",
            "保持熱情",
            "突破自我",
            "善於合作",
            "平衡發展",
            "樂觀積極",
            "系統規劃",
            "及時調整"
        ]
        return {
            "goal": random.choice(goals),
            "key_point": random.choice(key_points)
        }

    def generate_sentence(self):
        template = random.choice(list(self.template_mapping.keys()))
        data = self.template_mapping[template]()
        try:
            return template.format(**data)
        except KeyError as e:
            print(f"Error with template: {template}")
            print(f"Data: {data}")
            raise e
        
    def generate_text_by_type(self, text_type):
        """根據類型生成文本，確保所有佔位符都有對應的數據"""
        if text_type not in self.template_groups:
            text_type = random.choice(list(self.template_groups.keys()))
            
        template = random.choice(self.template_groups[text_type])
        
        # 準備所有可能的填充數據
        data = {
            # 基礎描述詞
            'adj': random.choice(['美好', '溫暖', '寧靜', '熱鬧', '生機勃勃', '平和']),
            'time': random.choice(self.words['time']),
            'location': random.choice(self.words['location']),
            'scene': random.choice(self.words['scene']),
            'feeling': random.choice(self.words['feeling']),
            'result': random.choice(['讓人心曠神怡', '令人陶醉', '給人深刻印象', '使人回味無窮']),
            
            # 人物相關
            'person_a': random.choice(['我', '他', '她', '老師', '同學', '朋友']),
            'person_b': random.choice(['小明', '小華', '老李', '張老師', '王同學']),
            'reaction': random.choice(['露出了笑容', '點點頭', '若有所思', '深表贊同', '認真思考']),
            'dialogue': random.choice([
                '努力總會有收穫的',
                '讓我們一起加油吧',
                '這確實需要認真思考',
                '生活中處處有學問',
                '堅持就是勝利'
            ]),
            'dialogue_scene': random.choice([
                '氣氛很是融洽',
                '各抒己見',
                '達成了共識',
                '收穫頗豐',
                '相談甚歡'
            ]),
            
            # 議論相關
            'topic': random.choice(self.words['topic']),
            'opinion': random.choice([
                '需要持續創新',
                '應該與時俱進',
                '重在堅持不懈',
                '關鍵在於平衡',
                '必須腳踏實地'
            ]),
            'reason': random.choice([
                '這是發展的必然趨勢',
                '這樣才能達到最好的效果',
                '這符合大多數人的需求',
                '這是經過實踐證明的',
                '這能帶來持續的進步'
            ]),
            'aspect': random.choice(['整體', '長遠', '多元', '創新', '實踐']),
            'analysis': random.choice([
                '需要更多的思考和探索',
                '值得我們深入研究',
                '還有很大的發展空間',
                '已經顯現出重要價值',
                '將帶來深遠的影響'
            ]),
            'point': random.choice([
                '方法的創新',
                '理念的轉變',
                '實踐的重要性',
                '持續的投入',
                '系統的思考'
            ]),
            'goal': random.choice([
                '實現更好的發展',
                '達到預期的效果',
                '獲得理想的成果',
                '推動持續進步',
                '創造更多價值'
            ]),
            
            # 記敘相關
            'event': random.choice([
                '第一次參加比賽',
                '遇到了困難',
                '完成了一個項目',
                '經歷了一次挑戰',
                '達成了一個目標'
            ]),
            'insight': random.choice([
                '堅持的重要性',
                '團隊的力量',
                '創新的價值',
                '學習的必要',
                '溝通的藝術'
            ]),
            'detail': random.choice([
                '那些努力的時刻',
                '與夥伴合作的經歷',
                '克服困難的過程',
                '收穫成果的喜悅',
                '得到認可的瞬間'
            ]),
            
            # 科學相關
            'field': random.choice(list(self.science_terms.keys())),
            'research': random.choice([
                '最新研究',
                '實驗數據',
                '觀察結果',
                '理論分析',
                '系統研究'
            ]),
            'theory': random.choice([
                '量子理論',
                '相對論',
                '演化論',
                '系統論',
                '控制論'
            ]),
            'situation': random.choice([
            '困難', '挑戰', '機遇', '變化', '成功', '失敗',
            '學習', '工作', '生活', '休息', '思考', '探索'
            ]),
            'emotion': random.choice(self.words['emotion']),
            'thought': random.choice([
                '人生的意義', '生活的真諦', '成長的過程',
                '夢想的追求', '目標的實現', '價值的創造'
            ]),
            'action': random.choice([
                '思考', '回憶', '憧憬', '計劃', '行動',
                '學習', '創造', '分享', '探索', '嘗試'
            ]),
            'concept': random.choice([
                '能量', '物質', '時間', '空間', '結構',
                '系統', '模式', '關係', '過程', '規律'
            ]),
            'foundation': random.choice([
                '基本原理', '核心概念', '科學發現',
                '實驗驗證', '理論推導', '數學模型'
            ]),
            'principle': random.choice([
                '守恆定律', '相對性原理', '對稱性原理',
                '因果關係', '演化規律', '系統理論'
            ]),
            'task': random.choice([
                '研究', '開發', '創新', '改進', '優化',
                '分析', '設計', '實現', '驗證', '應用'
            ]),
            'phenomenon': random.choice([
                "量子糾纏", "超導現象", "光電效應", "引力波動", 
                "化學反應", "生物演化", "天體運動", "能量轉換",
                "物質變化", "熱力擴散", "電磁感應", "核裂變"
            ]),
            
            'memory': random.choice([
                "童年時光", "學習歷程", "工作經驗", "生活點滴",
                "旅行記憶", "成長故事", "重要時刻", "難忘經歷",
                "美好回憶", "寶貴教訓", "人生轉折", "感動瞬間"
            ]),
            'industry': random.choice([
                "製造業", "服務業", "科技業", "金融業", "教育業",
                "醫療業", "能源業", "運輸業", "零售業", "建築業",
                "農業", "環保業", "文創產業", "旅遊業", "通訊業"
            ]),
            
            'problem': random.choice([
                "技術瓶頸", "資源短缺", "效率低下", "成本過高",
                "品質問題", "環境污染", "能源消耗", "安全隱患",
                "管理難題", "系統缺陷", "市場競爭", "人才流失"
            ]),
            
            'law': random.choice([
                "能量守恆定律", "質量守恆定律", "熱力學定律",
                "運動定律", "電磁定律", "進化定律", "市場規律",
                "供需法則", "經濟規律", "社會發展規律"
            ]),
            
            'experiment': random.choice([
                "實驗研究", "科學實驗", "對照試驗", "田野調查",
                "臨床試驗", "模擬實驗", "測試驗證", "探索性研究",
                "應用研究", "基礎研究", "創新試驗", "驗證實驗"
            ]),
            
            'method': random.choice([
                "系統分析", "實證研究", "統計分析", "模型模擬",
                "理論推導", "實驗驗證", "田野調查", "案例研究",
                "比較研究", "跨學科研究", "定性分析", "定量研究"
            ]),
            'perspective': random.choice([
                "理論視角", "實踐層面", "技術觀點", "系統思維",
                "創新思路", "發展前景", "科學角度", "應用方向",
                "研究方法", "整體規劃", "戰略高度", "細節關注"
            ]),
            
            'data': random.choice([
                "實驗數據", "研究資料", "統計結果", "調查報告",
                "觀測記錄", "測量結果", "分析報告", "檢測數據",
                "監測資料", "評估結果", "市場數據", "用戶反饋"
            ]),
            
            'observation': random.choice([
                "實驗觀察", "實地考察", "現場記錄", "數據監測",
                "過程跟蹤", "現象描述", "行為分析", "變化記錄",
                "規律發現", "特徵總結", "趨勢觀測", "效果評估"
            ]),
            
            'technology': random.choice([
                "人工智能", "大數據分析", "雲計算技術", "物聯網",
                "區塊鏈", "5G通訊", "量子計算", "虛擬現實",
                "增強現實", "機器學習", "深度學習", "自然語言處理"
            ]),
            
            'condition': random.choice([
                "特定條件", "實驗環境", "研究背景", "應用場景",
                "技術要求", "操作規範", "限制因素", "邊界條件",
                "初始狀態", "約束條件", "控制變量", "影響因素"
            ]),
            
            'subject': random.choice([
                "研究對象", "實驗主體", "分析目標", "觀察重點",
                "關注焦點", "探討課題", "研究內容", "測試項目",
                "評估主題", "發展方向", "創新領域", "應用範疇"
            ]),
            'phenomenon': random.choice([
                "量子糾纏", "超導現象", "光電效應", "引力波動", 
                "化學反應", "生物演化", "天體運動", "能量轉換",
                "物質變化", "熱力擴散", "電磁感應", "核裂變"
            ]),
            
            'memory': random.choice([
                "童年時光", "學習歷程", "工作經驗", "生活點滴",
                "旅行記憶", "成長故事", "重要時刻", "難忘經歷",
                "美好回憶", "寶貴教訓", "人生轉折", "感動瞬間"
            ]),
            
            'industry': random.choice([
                "製造業", "服務業", "科技業", "金融業", "教育業",
                "醫療業", "能源業", "運輸業", "零售業", "建築業",
                "農業", "環保業", "文創產業", "旅遊業", "通訊業"
            ]),
            
            'problem': random.choice([
                "技術瓶頸", "資源短缺", "效率低下", "成本過高",
                "品質問題", "環境污染", "能源消耗", "安全隱患",
                "管理難題", "系統缺陷", "市場競爭", "人才流失"
            ]),
            
            'law': random.choice([
                "能量守恆定律", "質量守恆定律", "熱力學定律",
                "運動定律", "電磁定律", "進化定律", "市場規律",
                "供需法則", "經濟規律", "社會發展規律"
            ]),
            
            'experiment': random.choice([
                "實驗研究", "科學實驗", "對照試驗", "田野調查",
                "臨床試驗", "模擬實驗", "測試驗證", "探索性研究",
                "應用研究", "基礎研究", "創新試驗", "驗證實驗"
            ]),
            
            'method': random.choice([
                "系統分析", "實證研究", "統計分析", "模型模擬",
                "理論推導", "實驗驗證", "田野調查", "案例研究",
                "比較研究", "跨學科研究", "定性分析", "定量研究"
            ]),
            'characteristic': random.choice([
                "穩定性", "可靠性", "創新性", "適應性", "實用性",
                "先進性", "高效性", "系統性", "靈活性", "前瞻性",
                "普遍性", "獨特性", "周期性", "持續性", "互補性"
            ]),
            
            'trend': random.choice([
                "數字化轉型", "智能化升級", "綠色發展", "創新驅動",
                "融合發展", "協同創新", "精細管理", "質量提升",
                "效能優化", "產業升級", "技術革新", "模式創新"
            ]),
            
            'factor': random.choice([
                "技術因素", "資源條件", "環境影響", "市場需求",
                "政策導向", "成本效益", "人才儲備", "創新能力",
                "競爭優勢", "發展潛力", "風險管控", "管理水平"
            ]),
            'pattern': random.choice([
                "發展模式", "運行規律", "變化趨勢", "演化路徑",
                "分布特徵", "互動方式", "組織形式", "結構模型",
                "循環規律", "生長模式", "傳播路徑", "演進規律",
                "關聯模式", "擴散模型", "適應機制", "創新模式"
            ])
        }
        
        # 如果是科學類型，添加特定領域的術語
        if text_type == '科學':
            field_data = self.generate_scientific_content(data['field'])
            data.update(field_data)
        elif text_type == '技術':
            tech_data = self.generate_technical_content()
            data.update(tech_data)
            
        try:
            return template.format(**data)
        except KeyError as e:
            #print(f"Missing key in template: {template}")
            print(f"Missing key: {e}")
            #print(f"Available keys: {data.keys()}")
            return "生活中充滿了驚喜和期待。"

    def generate_paragraph(self, min_sentences=3, max_sentences=5):
        """生成段落，包含多個句子"""
        num_sentences = random.randint(min_sentences, max_sentences)
        text_types = list(self.template_groups.keys())
        sentences = []
        
        # 確保段落中的句子類型有連貫性
        current_type = random.choice(text_types)
        sentences.append(self.generate_text_by_type(current_type))
        
        for _ in range(num_sentences - 1):
            # 有80%的概率保持同一類型
            if random.random() < 0.8:
                sentences.append(self.generate_text_by_type(current_type))
            else:
                current_type = random.choice(text_types)
                sentences.append(self.generate_text_by_type(current_type))
        
        return ''.join(sentences)

    def generate_dataset(self, num_samples=100000, min_sentences=3, max_sentences=5):
        """生成訓練數據集"""
        dataset = []
        for i in range(num_samples):
            try:
                text = self.generate_paragraph(min_sentences, max_sentences)
                dataset.append({"text": text})
                
                if (i + 1) % 10000 == 0:
                    print(f"已生成 {i + 1} 條數據")
                    
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue
            
        output_file = "training_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        return dataset

def generate_training_data():
    """生成完整的訓練數據集"""
    generator = ChineseDataGenerator()
    
    # 生成大量數據
    print("開始生成訓練數據...")
    dataset = generator.generate_dataset(
        num_samples=100000,   # 增加到100000條
        min_sentences=3,      # 每條數據至少3個句子
        max_sentences=5       # 最多5個句子
    )
    
    print(f"成功生成 {len(dataset)} 條訓練數據")
    return dataset

# 使用示例
if __name__ == "__main__":
    generator = ChineseDataGenerator()
    dataset = generator.generate_dataset(100000)
    print(f"已生成{len(dataset)}條訓練數據")