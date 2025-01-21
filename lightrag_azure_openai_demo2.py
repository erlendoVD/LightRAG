import asyncio
import tkinter as tk
from tkinter import scrolledtext

async def test_funcs():
    result = await llm_model_func("How are you?")
    print("Resposta do llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("Resultado do embedding_func: ", result.shape)
    print("Dimensão da embedding: ", result.shape[1])

asyncio.run(test_funcs())

embedding_dimension = 3072

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=embedding_dimension,
        max_token_size=8192,
        func=embedding_func,
    ),
)

book1 = open("./data/Anbud1.txt", encoding="utf-8")
book2 = open("./data/Anbud2.txt", encoding="utf-8")

rag.insert([book1.read(), book2.read()])

def query_rag():
    query_text = query_entry.get()
    result_naive = rag.query(query_text, param=QueryParam(mode="naive"))
    result_local = rag.query(query_text, param=QueryParam(mode="local"))
    
    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Result (Naive):\n{result_naive}\n\nResult (Local):\n{result_local}")
    result_text.config(state=tk.DISABLED)

# Create the main window
root = tk.Tk()
root.title("LightRAG Query Interface")

# Create a text entry for the query
query_label = tk.Label(root, text="Enter your query:")
query_label.pack(pady=5)

query_entry = tk.Entry(root, width=50)
query_entry.pack(pady=5)

# Create a button to submit the query
query_button = tk.Button(root, text="Submit", command=query_rag)
query_button.pack(pady=5)

# Create a scrolled text widget to display the results
result_text = scrolledtext.ScrolledText(root, width=60, height=20, state=tk.DISABLED)
result_text.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()