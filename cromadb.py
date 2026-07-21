import os
import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# ==========================================================
# 1. 원격 ChromaDB 연결 및 컬렉션 가져오기/생성하기 
# ==========================================================
client = chromadb.HttpClient(
    host="192.168.10.25",
    port=18000
)

collection_name = "guide_ver2"

# 백엔드(app.py)와 100% 일치하는 임베딩 함수 적용!
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="jhgan/ko-sroberta-multitask"
)

# 기존 컬렉션이 있으면 가져오고, 없으면 새로 생성합니다. 
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_func  
)
print(f"📦 '{collection_name}' 컬렉션에 연결되었습니다. (기존 데이터 유지)")


# ==========================================================
# 2. 파일 타입별 분할(Chunking) 함수 정의
# ==========================================================

# [함수 A] 마크다운 파일 분할 (표, ASCII 박스 및 [텍스트](URL) 링크 보존)
def split_markdown_file(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ 파일을 찾을 수 없어 건너뜁니다: {file_path}")
        return []
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 빈 줄(\n\n) 단위 분할
    paragraphs = content.split('\n\n')
    chunks = []
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
            
        # ASCII 박스선, 표 구조, 혹은 마크다운 링크([텍스트](URL))가 포함된 줄은 쪼개지 않고 통째로 보존합니다.
        if any(char in p for char in ["┌", "┐", "└", "┘", "│", "─", "|"]) or "http" in p:
            chunks.append(p)
            continue
            
        # 일반 본문 텍스트 분할 및 결합 (최소 문맥 유지를 위해 150자 내외 병합)
        sentences = re.split(r'(?<=[.!?])\s+', p)
        temp_chunk = ""
        for sentence in sentences:
            if len(temp_chunk) + len(sentence) < 150:
                temp_chunk += " " + sentence if temp_chunk else sentence
            else:
                if temp_chunk.strip():
                    chunks.append(temp_chunk.strip())
                temp_chunk = sentence
        if temp_chunk.strip():
            chunks.append(temp_chunk.strip())
            
    return chunks


# [함수 B] CSV 파일 분할 (행 단위로 의미 있게 문장화)
def split_csv_file(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ 파일을 찾을 수 없어 건너뜁니다: {file_path}")
        return []
        
    chunks = []
    df = pd.read_csv(file_path, encoding="utf-8")
    
    for idx, row in df.iterrows():
        # 각 행의 데이터를 챗봇이 이해하기 쉬운 텍스트 묶음으로 생성
        row_text = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        chunks.append(row_text)
        
    return chunks


# ==========================================================
# 3. 대상 파일 순차적으로 읽어서 DB에 삽입 (Upsert)
# ==========================================================
files_to_load = [
    {"path": r"C:\3ai\db\common_guide.md", "type": "md", "source": "common_guide"},
    {"path": r"C:\3ai\db\Eptatretus_burgeri.md", "type": "md", "source": "eptatretus_burgeri"},
    {"path": r"C:\3ai\db\Homarus_americanus.md", "type": "md", "source": "homarus_americanus"},
    {"path": r"C:\3ai\db\Mizuhopecten_yessoensis.md", "type": "md", "source": "mizuhopecten_yessoensis"},
    {"path": r"C:\3ai\db\unipass_guide.md", "type": "md", "source": "unipass_guide"},
    {"path": r"C:\3ai\db\quarantine_dataset.csv", "type": "csv", "source": "quarantine_dataset"}
]

for file_info in files_to_load:
    path = file_info["path"]
    file_type = file_info["type"]
    source_name = file_info["source"]
    
    print(f"\n[작업 시작] {source_name} 처리 중...")
    
    # 파일 타입에 따른 텍스트 분할 실행
    if file_type == "md":
        documents = split_markdown_file(path)
    elif file_type == "csv":
        documents = split_csv_file(path)
    else:
        documents = []
        
    if not documents:
        continue
        
    print(f"-> 분할 완료: {len(documents)}개의 청크 생성됨.")
    
    # 고유 ID 및 메타데이터 정의
    ids = [f"{source_name}_{i}" for i in range(len(documents))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(documents))]
    
    # DB에 적재 (Upsert 수행: ID가 동일하면 덮어쓰고, 없으면 추가)
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ {source_name} 데이터가 '{collection_name}' 컬렉션에 적재 완료되었습니다!")

print("\n🎉 모든 지정된 가이드 및 데이터셋 파일이 성공적으로 적재/갱신되었습니다!")