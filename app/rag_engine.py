import os, re
from dotenv import load_dotenv
from openai import OpenAI
from langchain.embeddings.base import Embeddings
from typing import List

class CustomOpenAIEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [embedding.embedding for embedding in resp.data]

    def embed_query(self, text: str) -> List[float]:
        resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
        return resp.data[0].embedding
from langchain_community.vectorstores import Chroma

load_dotenv()
client = OpenAI()

def _normalize_path_for_jmp(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")

def _sanitize_jsl(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"^```[\w-]*\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    text = text.strip().strip("'''").strip('"""').strip()
    text = text.replace("```JSL", "").replace("```jsl", "").replace("```", "")
    return text.strip()

def _looks_like_jsl(jsl: str) -> bool:
    if not jsl.strip():
        return False
    if jsl.count("(") != jsl.count(")"):
        return False
    return True

def load_vectordb(persist_dir: str = "app/rag_index"):
    embeddings = CustomOpenAIEmbeddings()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def retrieve_context(query: str, k: int = 4, persist_dir: str = "app/rag_index") -> str:
    db = load_vectordb(persist_dir)
    docs = db.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)

def prompt_to_jsl_rag(prompt: str, data_path: str, column_names: list[str], persist_dir: str = "app/rag_index") -> str:
    jmp_path = _normalize_path_for_jmp(data_path)
    retrieved = retrieve_context(prompt, k=4, persist_dir=persist_dir)

    system = f"""
You are an expert in JMP Scripting Language (JSL). Use ONLY valid, executable JSL.
Rules:
- Start with: dt = Open("{jmp_path}");
- Use only these columns: {', '.join(column_names)}
- Prefer Graph Builder, Distribution, Bivariate, Fit Model, Oneway when appropriate.
- End every statement with a semicolon.
- No markdown or explanations.

Relevant documentation snippets:
{retrieved}
    """.strip()

    # Try twice with slight temperature bump on retry
    for attempt in range(2):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.1 if attempt == 0 else 0.3,
        )
        raw = resp.choices[0].message.content or ""
        jsl = _sanitize_jsl(raw)

        # Guarantee Open() is correct and present
        if "Open(" not in jsl:
            jsl = f'dt = Open("{jmp_path}");\n' + jsl
        else:
            jsl = re.sub(r'Open\((["\']).*?\1\)', f'Open("{jmp_path}")', jsl, count=1, flags=re.IGNORECASE)

        if _looks_like_jsl(jsl):
            return jsl.strip()

        # retry with feedback
        prompt = f"Previous JSL failed or was incomplete. Fix and output only valid JSL:\n{jsl}"

    return "/* Failed to generate valid JSL after retries */"
