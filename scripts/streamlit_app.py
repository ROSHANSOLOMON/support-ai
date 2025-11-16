# streamlit_app.py
threads = st.slider("n_threads", 1, 12, 8)
max_tokens = st.slider("max tokens", 32, 512, 200)
question = st.text_input("Question", "How do I reset my password?")


# session-state holders
if "llm_obj" not in st.session_state:
st.session_state.llm_obj = None
if "retriever" not in st.session_state:
st.session_state.retriever = None


if st.button("(Re)load components"):
try:
st.info("Loading Retriever and LLM (this may take a little)...")
st.session_state.retriever = Retriever()
st.session_state.llm_obj = LLM(model_filename=model_file, n_threads=threads)
st.success("Components loaded.")
except Exception as e:
st.error(f"Failed to load: {e}")


if st.session_state.llm_obj is None or st.session_state.retriever is None:
st.warning("Click '(Re)load components' to initialize the Retriever and LLM.")
st.stop()


if st.button("Ask (run)"):
if not question.strip():
st.warning("Please type a question.")
else:
t0 = time.time()
with st.spinner("Retrieving relevant documents..."):
r = st.session_state.retriever
docs = r.query(question, k=4)
context_texts = []
for d in docs[:3]:
t = d.get("text", "") or ""
src = d.get("source", "unknown")
context_texts.append(f"Source: {src}\n{t}\n")
enriched = "\n\n".join(context_texts)


with st.spinner("Generating answer from local LLM..."):
llm = st.session_state.llm_obj
ans = llm(question, enriched)
elapsed = time.time() - t0


st.subheader("==== ANSWER ====")
st.write(ans)
st.markdown("---")
st.subheader("Top sources")
for i, d in enumerate(docs[:6]):
st.write(f"**[{i}]** {d.get('source')} (score={d.get('score')})")
st.write(d.get("text","")[:800] + ("..." if len(d.get("text",""))>800 else ""))
st.caption(f"Elapsed: {elapsed:.2f} s (retrieve + generation)")