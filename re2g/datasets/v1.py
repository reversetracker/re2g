import abc
import random

import torch
from datasets import load_dataset
from datasets.utils.typing import PathLike
from langchain_community.retrievers import BM25Retriever as BM25
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer

from re2g.configs import settings

PRETRAINED_MODEL_NAME_OR_PATH = settings.pretrained_model_name_or_path

DATALOADER_NUM_WORKERS = settings.dataloader_num_workers

QUERY_MAX_LENGTH = settings.query_max_length

QUERY_PADDING = settings.query_padding

CONTEXT_MAX_LENGTH = settings.context_max_length

CONTEXT_PADDING = settings.context_padding

RERANK_MAX_LENGTH = settings.rerank_max_length

RERANK_PADDING = settings.rerank_padding


class QueryEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        return [[random.uniform(-1, 1) for _ in range(128)] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return [random.uniform(-1, 1) for _ in range(128)]


class ContextEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        return [[random.uniform(-1, 1) for _ in range(128)] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return [random.uniform(-1, 1) for _ in range(128)]


class Retriever(abc.ABC):

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.context_documents)

    @property
    def context_documents(self) -> list[Document]:
        documents = []
        done = set()
        for item in self.dataset:
            title, context, metadata = (
                item["title"],
                item["context"],
                {"title": item["title"]},
            )
            if title in done:
                continue
            documents.append(Document(page_content=context, metadata=metadata))
            done.add(title)
        return documents

    def fetch(self, query: str, k: int = 64) -> list[Document]:
        raise NotImplementedError

    def fetch_by_vector(self, query: list[float], k: int = 64) -> list[Document]:
        raise NotImplementedError


class BM25Retriever(Retriever):
    def __init__(
        self,
        dataset: Dataset,
        pretrained_model_name_or_path: str | PathLike = None,
    ):
        super().__init__(dataset)
        self.tokenizer = ElectraTokenizer.from_pretrained(
            pretrained_model_name_or_path or "monologg/koelectra-base-v3-discriminator"
        )
        self.bm25 = BM25.from_documents(
            self.context_documents,
            preprocess_func=self.tokenizer.tokenize,
        )

    def fetch(self, query: str, k: int = 64) -> list[Document]:
        self.bm25.k = k
        return self.bm25.get_relevant_documents(query)

    def fetch_by_vector(self, query: list[float], k: int = 64) -> list[Document]:
        raise NotImplementedError(
            "BM25Retriever does not support vector-based retrieval."
        )


class DprRetriever(Retriever):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self._query_embeddings = QueryEmbeddings()
        self._context_embeddings = ContextEmbeddings()
        self._chroma = Chroma.from_documents(
            self.context_documents, self._context_embeddings
        )

    def fetch(self, query: str, k: int = 64) -> list[Document]:
        vector = self._query_embeddings.embed_query(query)
        return self.fetch_by_vector(vector, k=k)

    def fetch_by_vector(self, query: list[float], k: int = 64) -> list[Document]:
        return self._chroma.similarity_search_by_vector(query, k=k)


class SquadDataset(Dataset):
    def __init__(self, split: str = "train"):
        self.dataset = load_dataset("squad_kor_v1", split=split)

    def shuffle(self, seed: int = 69):
        self.dataset = self.dataset.shuffle(seed=seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[idx]


class DprDataset(Dataset):
    def __init__(self, dataset: Dataset, bm25_k: int = 64):
        self.dataset = dataset
        self.bm25_k = bm25_k
        self.bm25 = BM25Retriever(self.dataset)

    @staticmethod
    def example():
        """Data 어떻게 생긴지 궁금 했잖아.. 그래서 만들어 봤어, 맞다면 허공에 '앙 기모찌!' 라고 외쳐."""
        return {
            "id": "6521755-3-1",
            "title": "알렉산더_헤이그",
            "context": '그의 편에 헤이그는 지구촌의 논점들의 국내적 정치 노력들에 관해서만 근심한 레이건의 가까운 조언자들을 "외교 정책의 아마추어"로 묘사하였다. 1982년 6월 25일 결국적으로 온 그의 국무장관으로서 사임은 불가능한 상황이 된 것을 끝냈다. 헤이그는 개인적 생활로 돌아갔다가 1988년 대통령 선거를 위한 공화당 후보직을 안정시키는 시도를 하는 데 충분하게 정계로 돌아갔으나 후보직을 이기는 데 성원을 가지지 않았다. 그는 외교 정책 논쟁들에 연설자로서 활동적으로 남아있었으나 그의 전념은 정치에서 개인적 생활로 옮겨졌다. 그는 Worldwide Associates Inc.의 국제적 상담 회사에 의하여 기용되었고, 그 기구의 의장과 회장이 되었다.',
            "question": "헤이그가 사적생활을 하다가 정계로 돌아갔던 해는 언제인가?",
            "answers": {"text": ["1988년"], "answer_start": [153]},
            "bm25_contexts": [
                "사법연수원을 16기로 수료한 후 변호사 생활을 하다가 16대 총선에서 서울 강남을에 출마하여 정계에 입문했다. 제16대 국회의원 시절 4년 연속 시민단체 주관 국정감사 우수위원으로 선정되었고, 정치개혁특위 간사를 맡아 깨끗하고 투명한 선거를 목적으로 한 소위 ‘오세훈 선거법’으로 불리는 3개 정치관계법 개정을 주도했다. 2006년 서울시장에 당선되어 2011년까지 서울특별시장을 연임하며 창의시정과 디자인 서울을 주요 정책으로 하면서, 청렴도 향상, 강남북 균형발전, 복지 정책 희망드림 프로젝트, 대기환경 개선 등에 주력하였고, 다산 콜 센터와 장기전세주택 시프트를 도입하였다. 2011년 저소득층을 대상으로 선별적 복지를 주장하며 서울시 무상 급식 정책에서 주민 투표를 제안하고, 투표율이 미달되자 시장직을 사퇴하였다. 바른정당 상임고문을 지내다가 국민의당과의 합당에 반대하며 2018년 2월 5일 바른정당을 탈당했다.",
                "지원 이동통신의 경우, LTE Cat.12·13, LTE Cat.9 그리고 LTE Cat.6 모델이 있다. 우선, 업로드 속도는 Cat.13이 150 Mbps, Cat.9와 Cat.6이 50 Mbps로 최대 속도가 잡혀있고, 다운로드 속도는 Cat.12가 600 Mbps, Cat.9가 450 Mbps 그리고 Cat.6이 300 Mbps로 최대 속도가 잡혀져있다. 3 Band 캐리어 어그리게이션의 경우 상황에 따라 추가적으로 지원하며, VoLTE를 지원한다. 또한, 갤럭시 S7과 같이 모든 기기의 통신 모뎀 솔루션이 모바일 AP에 내장된 최초의 갤럭시 S 시리즈 중 하나이다. 이는 퀄컴 스냅드래곤 시리즈를 탑재한 기존 갤럭시 S 시리즈 스마트폰은 극소수를 제외하면 통신 모뎀 솔루션이 기본적으로 내장되어 있었으나, 삼성 엑시노스 시리즈는 플래그십 AP로는 삼성 엑시노스 8890이 통신 모뎀 솔루션을 내장한 최초의 모바일 AP이기 때문이다.",
                "대학로에서 소문난 연기파 배우로서 2004년 아카펠라 연극 '겨울공주 평강이야기'를 시작으로 연극과 뮤지컬 무대에서 입지를 다졌다. 2015년 드라마 《육룡이 나르샤》에서 정도전의 혁명동지 역으로 시청자들에게 눈도장을 찍었으며, 2017년 영화 《범죄도시》에서는 흑룡파 조직의 보스 장첸(윤계상)의 오른팔로서 삭발한 머리와 날카로운 눈빛으로 등장하여 첫 악역 연기에 선보였다. 진선규는 600만 관객을 동원한 영화 《범죄도시》를 \"연기 인생의 터닝 포인트이자 인생작\"이라고 말했다. 진선규는 이 영화로 2017년 청룡영화제 남우조연상을 수상하였고, 시상식에서 수상자로 호명돼 무대에 오르자마자 눈물을 쏟는 수상 소감으로 감동을 안겼다.",
            ],
            "bm25_labels": [0, 0, 0],
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        title, context, query = item["title"], item["context"], item["question"]

        # collecting bm25
        bm25_documents = self.bm25.fetch(query, k=self.bm25_k + 1)
        bm25_documents = [x for x in bm25_documents if title != x.metadata["title"]]
        bm25_documents = bm25_documents[: self.bm25_k]
        bm25_contexts = [x.page_content for x in bm25_documents]
        bm25_labels = [int(title == x.metadata["title"]) for x in bm25_documents]

        # adding bm25 to the item
        item["bm25_contexts"] = bm25_contexts
        item["bm25_labels"] = bm25_labels
        return item


class DprDataModule(LightningDataModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str | PathLike,
        batch_size: int = 128,
        bm25_k: int = 64,
        seed: int = 69,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.bm25_k = bm25_k
        self.tokenizer = ElectraTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.seed = seed

        self.train_squad_dataset = SquadDataset(split="train")
        self.train_dataset = DprDataset(self.train_squad_dataset, bm25_k=self.bm25_k)

        self.val_squad_dataset = SquadDataset(split="validation")
        self.val_squad_dataset.shuffle(seed=self.seed)
        self.val_dataset = DprDataset(self.val_squad_dataset, bm25_k=self.bm25_k)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=DATALOADER_NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=DATALOADER_NUM_WORKERS,
        )

    def _collate_fn(self, batch: list[dict]):
        queries = [x["question"] for x in batch]
        contexts = [x["context"] for x in batch]
        bm25_contexts_list = [x["bm25_contexts"] for x in batch]
        bm25_labels_list = [x["bm25_labels"] for x in batch]

        query_batch_encoding = self.tokenizer.batch_encode_plus(
            queries,
            max_length=QUERY_MAX_LENGTH,
            padding=QUERY_PADDING,
            truncation=True,
            return_tensors="pt",
        )
        context_batch_encoding = self.tokenizer.batch_encode_plus(
            contexts,
            max_length=CONTEXT_MAX_LENGTH,
            padding=CONTEXT_PADDING,
            truncation=True,
            return_tensors="pt",
        )

        bm25_batch_encodings = []
        for bm25_contexts in bm25_contexts_list:
            bm25_contexts_encoding = self.tokenizer.batch_encode_plus(
                bm25_contexts,
                max_length=CONTEXT_MAX_LENGTH,
                padding=CONTEXT_PADDING,
                truncation=True,
                return_tensors="pt",
            )
            bm25_batch_encodings.append(bm25_contexts_encoding)

        bm25_input_ids = torch.stack(
            [x["input_ids"] for x in bm25_batch_encodings], dim=0
        )
        bm25_attention_mask = torch.stack(
            [x["attention_mask"] for x in bm25_batch_encodings], dim=0
        )
        bm25_token_type_ids = torch.stack(
            [x["token_type_ids"] for x in bm25_batch_encodings], dim=0
        )
        bm25_labels = torch.stack([torch.tensor(x) for x in bm25_labels_list], dim=0)
        return {
            "query_input_ids": query_batch_encoding["input_ids"],
            "query_attention_mask": query_batch_encoding["attention_mask"],
            "query_token_type_ids": query_batch_encoding["token_type_ids"],
            "queries": queries,
            "context_input_ids": context_batch_encoding["input_ids"],
            "context_attention_mask": context_batch_encoding["attention_mask"],
            "context_token_type_ids": context_batch_encoding["token_type_ids"],
            "contexts": contexts,
            "bm25_input_ids": bm25_input_ids,
            "bm25_attention_mask": bm25_attention_mask,
            "bm25_token_type_ids": bm25_token_type_ids,
            "bm25_contexts": bm25_contexts_list,
            "bm25_labels": bm25_labels,
        }


class RerankDataset(Dataset):
    def __init__(self, dataset: Dataset, bm25_k: int = 64, dpr_k: int = 64):
        self.bm25_k = bm25_k
        self.dpr_k = dpr_k
        self.dataset = dataset
        self.bm25 = BM25Retriever(self.dataset)
        self.dpr = DprRetriever(self.dataset)

    @staticmethod
    def example():
        """Data 어떻게 생긴지 궁금 했잖아.. 그래서 만들어 봤어, 맞다면 허공에 '앙 기모찌!' 라고 외쳐."""
        return {
            "id": "6521755-3-1",
            "title": "알렉산더_헤이그",
            "context": '그의 편에 헤이그는 지구촌의 논점들의 국내적 정치 노력들에 관해서만 근심한 레이건의 가까운 조언자들을 "외교 정책의 아마추어"로 묘사하였다. 1982년 6월 25일 결국적으로 온 그의 국무장관으로서 사임은 불가능한 상황이 된 것을 끝냈다. 헤이그는 개인적 생활로 돌아갔다가 1988년 대통령 선거를 위한 공화당 후보직을 안정시키는 시도를 하는 데 충분하게 정계로 돌아갔으나 후보직을 이기는 데 성원을 가지지 않았다. 그는 외교 정책 논쟁들에 연설자로서 활동적으로 남아있었으나 그의 전념은 정치에서 개인적 생활로 옮겨졌다. 그는 Worldwide Associates Inc.의 국제적 상담 회사에 의하여 기용되었고, 그 기구의 의장과 회장이 되었다.',
            "question": "헤이그가 사적생활을 하다가 정계로 돌아갔던 해는 언제인가?",
            "answers": {"text": ["1988년"], "answer_start": [153]},
            "bm25_contexts": [
                "사법연수원을 16기로 수료한 후 변호사 생활을 하다가 16대 총선에서 서울 강남을에 출마하여 정계에 입문했다. 제16대 국회의원 시절 4년 연속 시민단체 주관 국정감사 우수위원으로 선정되었고, 정치개혁특위 간사를 맡아 깨끗하고 투명한 선거를 목적으로 한 소위 ‘오세훈 선거법’으로 불리는 3개 정치관계법 개정을 주도했다. 2006년 서울시장에 당선되어 2011년까지 서울특별시장을 연임하며 창의시정과 디자인 서울을 주요 정책으로 하면서, 청렴도 향상, 강남북 균형발전, 복지 정책 희망드림 프로젝트, 대기환경 개선 등에 주력하였고, 다산 콜 센터와 장기전세주택 시프트를 도입하였다. 2011년 저소득층을 대상으로 선별적 복지를 주장하며 서울시 무상 급식 정책에서 주민 투표를 제안하고, 투표율이 미달되자 시장직을 사퇴하였다. 바른정당 상임고문을 지내다가 국민의당과의 합당에 반대하며 2018년 2월 5일 바른정당을 탈당했다.",
                "지원 이동통신의 경우, LTE Cat.12·13, LTE Cat.9 그리고 LTE Cat.6 모델이 있다. 우선, 업로드 속도는 Cat.13이 150 Mbps, Cat.9와 Cat.6이 50 Mbps로 최대 속도가 잡혀있고, 다운로드 속도는 Cat.12가 600 Mbps, Cat.9가 450 Mbps 그리고 Cat.6이 300 Mbps로 최대 속도가 잡혀져있다. 3 Band 캐리어 어그리게이션의 경우 상황에 따라 추가적으로 지원하며, VoLTE를 지원한다. 또한, 갤럭시 S7과 같이 모든 기기의 통신 모뎀 솔루션이 모바일 AP에 내장된 최초의 갤럭시 S 시리즈 중 하나이다. 이는 퀄컴 스냅드래곤 시리즈를 탑재한 기존 갤럭시 S 시리즈 스마트폰은 극소수를 제외하면 통신 모뎀 솔루션이 기본적으로 내장되어 있었으나, 삼성 엑시노스 시리즈는 플래그십 AP로는 삼성 엑시노스 8890이 통신 모뎀 솔루션을 내장한 최초의 모바일 AP이기 때문이다.",
                "대학로에서 소문난 연기파 배우로서 2004년 아카펠라 연극 '겨울공주 평강이야기'를 시작으로 연극과 뮤지컬 무대에서 입지를 다졌다. 2015년 드라마 《육룡이 나르샤》에서 정도전의 혁명동지 역으로 시청자들에게 눈도장을 찍었으며, 2017년 영화 《범죄도시》에서는 흑룡파 조직의 보스 장첸(윤계상)의 오른팔로서 삭발한 머리와 날카로운 눈빛으로 등장하여 첫 악역 연기에 선보였다. 진선규는 600만 관객을 동원한 영화 《범죄도시》를 \"연기 인생의 터닝 포인트이자 인생작\"이라고 말했다. 진선규는 이 영화로 2017년 청룡영화제 남우조연상을 수상하였고, 시상식에서 수상자로 호명돼 무대에 오르자마자 눈물을 쏟는 수상 소감으로 감동을 안겼다.",
            ],
            "bm25_labels": [0, 0, 0],
            "dpr_contexts": [
                "사법연수원을 16기로 수료한 후 변호사 생활을 하다가 16대 총선에서 서울 강남을에 출마하여 정계에 입문했다. 제16대 국회의원 시절 4년 연속 시민단체 주관 국정감사 우수위원으로 선정되었고, 정치개혁특위 간사를 맡아 깨끗하고 투명한 선거를 목적으로 한 소위 ‘오세훈 선거법’으로 불리는 3개 정치관계법 개정을 주도했다. 2006년 서울시장에 당선되어 2011년까지 서울특별시장을 연임하며 창의시정과 디자인 서울을 주요 정책으로 하면서, 청렴도 향상, 강남북 균형발전, 복지 정책 희망드림 프로젝트, 대기환경 개선 등에 주력하였고, 다산 콜 센터와 장기전세주택 시프트를 도입하였다. 2011년 저소득층을 대상으로 선별적 복지를 주장하며 서울시 무상 급식 정책에서 주민 투표를 제안하고, 투표율이 미달되자 시장직을 사퇴하였다. 바른정당 상임고문을 지내다가 국민의당과의 합당에 반대하며 2018년 2월 5일 바른정당을 탈당했다.",
                "지원 이동통신의 경우, LTE Cat.12·13, LTE Cat.9 그리고 LTE Cat.6 모델이 있다. 우선, 업로드 속도는 Cat.13이 150 Mbps, Cat.9와 Cat.6이 50 Mbps로 최대 속도가 잡혀있고, 다운로드 속도는 Cat.12가 600 Mbps, Cat.9가 450 Mbps 그리고 Cat.6이 300 Mbps로 최대 속도가 잡혀져있다. 3 Band 캐리어 어그리게이션의 경우 상황에 따라 추가적으로 지원하며, VoLTE를 지원한다. 또한, 갤럭시 S7과 같이 모든 기기의 통신 모뎀 솔루션이 모바일 AP에 내장된 최초의 갤럭시 S 시리즈 중 하나이다. 이는 퀄컴 스냅드래곤 시리즈를 탑재한 기존 갤럭시 S 시리즈 스마트폰은 극소수를 제외하면 통신 모뎀 솔루션이 기본적으로 내장되어 있었으나, 삼성 엑시노스 시리즈는 플래그십 AP로는 삼성 엑시노스 8890이 통신 모뎀 솔루션을 내장한 최초의 모바일 AP이기 때문이다.",
                "대학로에서 소문난 연기파 배우로서 2004년 아카펠라 연극 '겨울공주 평강이야기'를 시작으로 연극과 뮤지컬 무대에서 입지를 다졌다. 2015년 드라마 《육룡이 나르샤》에서 정도전의 혁명동지 역으로 시청자들에게 눈도장을 찍었으며, 2017년 영화 《범죄도시》에서는 흑룡파 조직의 보스 장첸(윤계상)의 오른팔로서 삭발한 머리와 날카로운 눈빛으로 등장하여 첫 악역 연기에 선보였다. 진선규는 600만 관객을 동원한 영화 《범죄도시》를 \"연기 인생의 터닝 포인트이자 인생작\"이라고 말했다. 진선규는 이 영화로 2017년 청룡영화제 남우조연상을 수상하였고, 시상식에서 수상자로 호명돼 무대에 오르자마자 눈물을 쏟는 수상 소감으로 감동을 안겼다.",
            ],
            "dpr_labels": [0, 0, 0],
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        title, context, query = item["title"], item["context"], item["question"]

        bm25_documents = self.bm25.fetch(query, k=self.bm25_k)
        bm25_contexts = [x.page_content for x in bm25_documents]
        bm25_labels = [int(title == x.metadata["title"]) for x in bm25_documents]

        dpr_documents = self.dpr.fetch(query, k=self.dpr_k)
        dpr_contexts = [x.page_content for x in dpr_documents]
        dpr_labels = [int(title == x.metadata["title"]) for x in dpr_documents]

        item["bm25_contexts"] = bm25_contexts
        item["bm25_labels"] = bm25_labels
        item["dpr_contexts"] = dpr_contexts
        item["dpr_labels"] = dpr_labels
        return item


class RerankDataModule(LightningDataModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str | PathLike,
        batch_size: int = 128,
        bm25_k: int = 64,
        dpr_k: int = 64,
        seed: int = 69,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.bm25_k = bm25_k
        self.dpr_k = dpr_k
        self.tokenizer = ElectraTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.seed = seed

        # lazy loading
        self.train_squad_dataset = SquadDataset(split="train")
        self.train_rerank_dataset = RerankDataset(
            dataset=self.train_squad_dataset, bm25_k=self.bm25_k, dpr_k=self.dpr_k
        )

        self.val_squad_dataset = SquadDataset(split="validation")
        self.val_squad_dataset.shuffle(seed=self.seed)
        self.val_rerank_dataset = RerankDataset(
            dataset=self.val_squad_dataset, bm25_k=self.bm25_k, dpr_k=self.dpr_k
        )

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_rerank_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=DATALOADER_NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_rerank_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=DATALOADER_NUM_WORKERS,
        )

    def _collate_fn(self, batch: list[dict]):
        query_batch = [x["question"] for x in batch]
        contexts_batch = [
            [x["context"]] + x["bm25_contexts"] + x["dpr_contexts"] for x in batch
        ]
        labels_batch = [[1] + x["bm25_labels"] + x["dpr_labels"] for x in batch]

        assert len(query_batch) == len(contexts_batch) == len(labels_batch)

        encodings_batch = []
        for query, contexts in zip(query_batch, contexts_batch):
            bm25_contexts_encoding = self.tokenizer.batch_encode_plus(
                [(query, context) for context in contexts],
                max_length=RERANK_MAX_LENGTH,
                padding=RERANK_PADDING,
                truncation=True,
                return_tensors="pt",
            )
            encodings_batch.append(bm25_contexts_encoding)

        input_ids = torch.stack([x["input_ids"] for x in encodings_batch], dim=0)
        attention_mask = torch.stack(
            [x["attention_mask"] for x in encodings_batch], dim=0
        )
        token_type_ids = torch.stack(
            [x["token_type_ids"] for x in encodings_batch], dim=0
        )
        labels = torch.stack([torch.tensor(x) for x in labels_batch], dim=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
