from datasets import load_dataset


def test_klue_sts_dataset():
    # ['dp', 'mrc', 'ner', 'nli', 're', 'sts', 'wos', 'ynat']
    klue_dataset = load_dataset("klue", "sts")

    validation = klue_dataset["validation"][0]

    assert validation == {
        "guid": "klue-sts-v1_dev_00000",
        "source": "airbnb-rtt",
        "sentence1": "무엇보다도 호스트분들이 너무 친절하셨습니다.",
        "sentence2": "무엇보다도, 호스트들은 매우 친절했습니다.",
        "labels": {"label": 4.9, "real-label": 4.857142857142857, "binary-label": 1},
    }


def test_klue_mrc_dataset():
    # ['dp', 'mrc', 'ner', 'nli', 're', 'sts', 'wos', 'ynat']
    klue_dataset = load_dataset("klue", "mrc")

    validation = klue_dataset["validation"][0]

    assert validation == {
        "title": "BMW 코리아, 창립 25주년 기념 ‘BMW 코리아 25주년 에디션’ 한정 출시",
        "context": "BMW 코리아(대표 한상윤)는 창립 25주년을 기념하는 ‘BMW 코리아 25주년 에디션’을 한정 출시한다고 밝혔다. 이번 BMW 코리아 25주년 에디션(이하 25주년 에디션)은 BMW 3시리즈와 5시리즈, 7시리즈, 8시리즈 총 4종, 6개 모델로 출시되며, BMW 클래식 모델들로 선보인 바 있는 헤리티지 컬러가 차체에 적용돼 레트로한 느낌과 신구의 조화가 어우러진 차별화된 매력을 자랑한다. 먼저 뉴 320i 및 뉴 320d 25주년 에디션은 트림에 따라 옥스포드 그린(50대 한정) 또는 마카오 블루(50대 한정) 컬러가 적용된다. 럭셔리 라인에 적용되는 옥스포드 그린은 지난 1999년 3세대 3시리즈를 통해 처음 선보인 색상으로 짙은 녹색과 풍부한 펄이 오묘한 조화를 이루는 것이 특징이다. M 스포츠 패키지 트림에 적용되는 마카오 블루는 1988년 2세대 3시리즈를 통해 처음 선보인 바 있으며, 보랏빛 감도는 컬러감이 매력이다. 뉴 520d 25주년 에디션(25대 한정)은 프로즌 브릴리언트 화이트 컬러로 출시된다. BMW가 2011년에 처음 선보인 프로즌 브릴리언트 화이트는 한층 더 환하고 깊은 색감을 자랑하며, 특히 표면을 무광으로 마감해 특별함을 더했다. 뉴 530i 25주년 에디션(25대 한정)은 뉴 3시리즈 25주년 에디션에도 적용된 마카오 블루 컬러가 조합된다. 뉴 740Li 25주년 에디션(7대 한정)에는 말라카이트 그린 다크 색상이 적용된다. 잔잔하면서도 오묘한 깊은 녹색을 발산하는 말라카이트 그린 다크는 장식재로 활용되는 광물 말라카이트에서 유래됐다. 뉴 840i xDrive 그란쿠페 25주년 에디션(8대 한정)은 인도양의 맑고 투명한 에메랄드 빛을 연상케 하는 몰디브 블루 컬러로 출시된다. 특히 몰디브 블루는 지난 1993년 1세대 8시리즈에 처음으로 적용되었던 만큼 이를 오마주하는 의미를 담고 있다.",
        "news_category": "자동차",
        "source": "acrofan",
        "guid": "klue-mrc-v1_dev_01891",
        "is_impossible": False,
        "question_type": 2,
        "question": "말라카이트에서 나온 색깔을 사용한 에디션은?",
        "answers": {
            "answer_start": [666, 666],
            "text": ["뉴 740Li 25주년 에디션", "뉴 740Li 25주년"],
        },
    }


def test_klue_ynat_dataset():
    # ['dp', 'mrc', 'ner', 'nli', 're', 'sts', 'wos', 'ynat']
    klue_dataset = load_dataset("klue", "ynat")

    validation = klue_dataset["validation"][0]

    assert validation == {
        "guid": "ynat-v1_dev_00000",
        "title": "5억원 무이자 융자는 되고 7천만원 이사비는 안된다",
        "label": 2,
        "url": "https://news.naver.com/main/read.nhn?mode=LS2D&mid=shm&sid1=101&sid2=260&oid=001&aid=0009563542",
        "date": "2017.09.21. 오후 5:09",
    }
