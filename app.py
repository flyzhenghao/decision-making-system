import streamlit as st
import pandas as pd
import uuid
from src.core import KnowledgeBase, IngestionEngine, DecisionEngine

# Page Config
st.set_page_config(page_title="Decision Consultant", layout="wide", page_icon="🧠")

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Core Components
try:
    if 'kb' not in st.session_state:
        st.session_state.kb = KnowledgeBase()
    if 'ingestion' not in st.session_state:
        st.session_state.ingestion = IngestionEngine(st.session_state.kb)
    if 'decision' not in st.session_state:
        st.session_state.decision = DecisionEngine(st.session_state.kb)
except ValueError as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

# --- TRANSLATION CONFIG ---
if 'language' not in st.session_state:
    st.session_state.language = 'English'

translations = {
    'English': {
        'nav_library': 'Library',
        'nav_kb': 'Knowledge Base',
        'nav_consultant': 'Consultant',
        'library_title': '📚 Library',
        'kb_title': '🧠 Knowledge Base',
        'consultant_title': '🤖 Decision Consultant',
        'upload_text': 'Upload PDF or EPUB files to extract Principles and Cases.',
        'process_btn': 'Process Books',
        'tab_principles': 'Principles',
        'tab_cases': 'Cases',
        'tab_relationships': 'Relationships',
        'save_case': 'Save My Case',
        'get_advice': 'Get Advice',
        'input_placeholder': 'What problem or decision are you facing?',
        'advice_title': '### 💡 AI Advice',
        'save_success': 'Case saved to Knowledge Base!',
        'processing': 'Processing...',
        'done': 'Processing Complete!',
        'no_principles': 'No principles found. Go to the Library to upload books.',
        'no_cases': 'No cases found.',
        'edit_instr': 'You can edit the tables below. Click the trash icon to delete rows.',
        'rel_desc': 'Visualizing how cases map to principles (N:M).',
        'rel_grouped': '#### Grouped by Principle'
    },
    'Chinese': {
        'nav_library': '图书馆',
        'nav_kb': '知识库',
        'nav_consultant': '决策顾问',
        'library_title': '📚 图书馆',
        'kb_title': '🧠 知识库',
        'consultant_title': '🤖 决策顾问',
        'upload_text': '上传 PDF 或 EPUB 文件以提取原则和案例。',
        'process_btn': '处理书籍',
        'tab_principles': '原则',
        'tab_cases': '案例',
        'tab_relationships': '关系视图',
        'save_case': '保存我的案例',
        'get_advice': '获取建议',
        'input_placeholder': '你面临什么问题或决策？',
        'advice_title': '### 💡 AI 建议',
        'save_success': '案例已保存到知识库！',
        'processing': '处理中...',
        'done': '处理完成！',
        'no_principles': '未找到原则。请前往图书馆上传书籍。',
        'no_cases': '未找到案例。',
        'edit_instr': '你可以编辑下方的表格。点击垃圾桶图标删除行。',
        'rel_desc': '可视化案例与原则的映射关系 (N:M)。',
        'rel_grouped': '#### 按原则分组'
    }
}

t = translations[st.session_state.language]

# Sidebar
st.sidebar.title("🧠 Decision System")
st.session_state.language = st.sidebar.radio("Language / 语言", ['English', 'Chinese'])
t = translations[st.session_state.language] # Refresh t after selection

page = st.sidebar.radio("Navigation", [t['nav_library'], t['nav_kb'], t['nav_consultant']])

# --- LIBRARY PAGE ---
if page == t['nav_library']:
    st.title(t['library_title'])
    st.markdown("### Upload Books & Documents")
    st.write(t['upload_text'])
    
    uploaded_files = st.file_uploader("Choose files", type=["pdf", "epub"], accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} files selected.")
        
        if st.button(t['process_btn'], type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_files = len(uploaded_files)
            total_principles = 0
            total_cases = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"{t['processing']} {uploaded_file.name} ({i+1}/{total_files})...")
                
                try:
                    # 1. Extract Text
                    if uploaded_file.name.endswith(".pdf"):
                        text = st.session_state.ingestion.extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.name.endswith(".epub"):
                        with open(f"temp_{uploaded_file.name}", "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        text = st.session_state.ingestion.extract_text_from_epub(f"temp_{uploaded_file.name}")
                        import os
                        os.remove(f"temp_{uploaded_file.name}") 
                    else:
                        st.error(f"Unsupported file format: {uploaded_file.name}")
                        continue
                    
                    # 2. Extract Principles & Cases
                    stats = st.session_state.ingestion.extract_principles_and_cases(text, source_name=uploaded_file.name)
                    total_principles += stats["principles"]
                    total_cases += stats["cases"]
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                    if "insufficient_quota" in str(e) or "invalid_api_key" in str(e) or "429" in str(e) or "401" in str(e):
                        st.error("Critical API Error. Stopping process.")
                        break
                
                progress_bar.progress((i + 1) / total_files)
            
            status_text.text(t['done'])
            st.success(f"All done! Extracted {total_principles} principles and {total_cases} cases.")

# --- KNOWLEDGE BASE PAGE ---
elif page == t['nav_kb']:
    st.title(t['kb_title'])
    st.info(t['edit_instr'])
    
    tab1, tab2, tab3 = st.tabs([t['tab_principles'], t['tab_cases'], t['tab_relationships']])
    
    # Helper to get bilingual columns
    def get_cols(base_cols, lang):
        if lang == 'Chinese':
            return [c + "_cn" if c in ["summary", "description"] else c for c in base_cols]
        return base_cols

    with tab1:
        st.subheader(t['tab_principles'])
        principles = st.session_state.kb.get_all_principles()
        if principles:
            df_p = pd.DataFrame(principles)
            
            # Ensure new columns exist for old data
            if "parent_context" not in df_p.columns: df_p["parent_context"] = "General"
            
            # Select columns based on language
            # Reordered: Header (parent_context) -> Item (summary) -> Description -> Source
            cols = ["id", "parent_context", "summary", "description", "source"]
            if st.session_state.language == 'Chinese':
                # Fallback if _cn columns don't exist yet
                if "summary_cn" not in df_p.columns: df_p["summary_cn"] = df_p["summary"]
                if "description_cn" not in df_p.columns: df_p["description_cn"] = df_p["description"]
                cols = ["id", "parent_context", "summary_cn", "description_cn", "source"]
            
            # Editable Dataframe
            # Use dynamic key to force re-render on language change
            editor_key = f"editor_principles_{st.session_state.language}"
            
            edited_df_p = st.data_editor(
                df_p[cols],
                column_config={
                    "id": None, # Hide ID
                    "parent_context": st.column_config.TextColumn("Principles Header", width="medium"),
                    "summary": st.column_config.TextColumn("Principles Item", width="medium"),
                    "summary_cn": st.column_config.TextColumn("原则条目 (CN)", width="medium"),
                    "description": "Description",
                    "description_cn": "描述 (CN)",
                    "source": "Source"
                },
                use_container_width=True,
                num_rows="dynamic",
                key=editor_key
            )
            
            # Detect Changes
            if not df_p[cols].equals(edited_df_p):
                # Handle Deletions
                current_ids = set(edited_df_p["id"])
                original_ids = set(df_p["id"])
                deleted_ids = original_ids - current_ids
                for pid in deleted_ids:
                    st.session_state.kb.delete_principle(pid)
                
                # Handle Updates
                for index, row in edited_df_p.iterrows():
                    original_row = df_p[df_p['id'] == row['id']]
                    if not original_row.empty:
                        changed = False
                        full_record = original_row.iloc[0].to_dict()
                        for col in cols:
                            if row[col] != full_record.get(col):
                                full_record[col] = row[col]
                                changed = True
                        if changed:
                            st.session_state.kb.update_principle(full_record)
                st.rerun()
        else:
            st.info(t['no_principles'])
            
    with tab2:
        st.subheader(t['tab_cases'])
        cases = st.session_state.kb.get_all_cases()
        if cases:
            df_c = pd.DataFrame(cases)
            
            if "related_principle_summary" not in df_c.columns: df_c["related_principle_summary"] = ""
            if "description_cn" not in df_c.columns: df_c["description_cn"] = df_c["description"]
            
            cols = ["id", "description", "source", "type", "related_principle_summary"]
            if st.session_state.language == 'Chinese':
                cols = ["id", "description_cn", "source", "type", "related_principle_summary"]

            editor_key_c = f"editor_cases_{st.session_state.language}"
            
            edited_df_c = st.data_editor(
                df_c[cols],
                column_config={
                    "id": None,
                    "description": "Description",
                    "description_cn": "描述 (CN)",
                    "source": "Source",
                    "type": "Type",
                    "related_principle_summary": "Related Principle"
                },
                use_container_width=True,
                num_rows="dynamic",
                key=editor_key_c
            )

            if not df_c[cols].equals(edited_df_c):
                current_ids = set(edited_df_c["id"])
                original_ids = set(df_c["id"])
                deleted_ids = original_ids - current_ids
                for cid in deleted_ids:
                    st.session_state.kb.delete_case(cid)
                
                for index, row in edited_df_c.iterrows():
                    original_row = df_c[df_c['id'] == row['id']]
                    if not original_row.empty:
                        changed = False
                        full_record = original_row.iloc[0].to_dict()
                        for col in cols:
                            if row[col] != full_record.get(col):
                                full_record[col] = row[col]
                                changed = True
                        if changed:
                            st.session_state.kb.update_case(full_record)
                st.rerun()
        else:
            st.info(t['no_cases'])

    with tab3:
        st.subheader(t['tab_relationships'])
        
        # View Mode Toggle
        view_mode = st.radio("View Mode", ["Table", "Graph"], horizontal=True)
        
        principles = st.session_state.kb.get_all_principles()
        cases = st.session_state.kb.get_all_cases()
        
        if view_mode == "Table":
            st.write(t['rel_desc'])
            if cases:
                # Create a lookup for principle details
                p_lookup = {p["summary"]: p for p in principles}
                
                data = []
                for c in cases:
                    related_summary = c.get("related_principle_summary", "Unlinked")
                    
                    # Find parent context
                    parent = "General"
                    if related_summary in p_lookup:
                        parent = p_lookup[related_summary].get("parent_context", "General")
                    
                    # Handle Bilingual Display
                    p_header = parent
                    p_item = related_summary
                    
                    if st.session_state.language == 'Chinese':
                        # Try to find Chinese versions
                        if related_summary in p_lookup:
                            p_obj = p_lookup[related_summary]
                            p_header = p_obj.get("parent_context", "General") # Context might not have CN translation field yet, assume same or add later
                            p_item = p_obj.get("summary_cn", related_summary)
                    
                    data.append({
                        "Principles Header": p_header,
                        "Principles Item": p_item,
                        "Case": c["description"],
                        "Type": c.get("type", "Unknown"),
                        "Source": c["source"]
                    })
                
                df_rel = pd.DataFrame(data)
                st.markdown(t['rel_grouped'])
                st.dataframe(df_rel, use_container_width=True)
            else:
                st.info(t['no_cases'])
        
        else: # Graph View
            import graphviz
            if principles:
                graph = graphviz.Digraph()
                graph.attr(rankdir='LR')
                
                # Nodes for Books/Sources
                sources = set(p.get("source", "Unknown") for p in principles)
                for s in sources:
                    graph.node(s, s, shape='box', style='filled', color='lightblue')
                
                for p in principles:
                    p_label = p.get("summary_cn", p["summary"]) if st.session_state.language == 'Chinese' else p["summary"]
                    p_id = p["id"]
                    parent = p.get("parent_context", "General")
                    source = p.get("source", "Unknown")
                    
                    # Parent Node
                    parent_id = f"{source}_{parent}"
                    graph.node(parent_id, parent, shape='ellipse', style='filled', color='lightgrey')
                    graph.edge(source, parent_id)
                    
                    # Principle Node
                    graph.node(p_id, p_label, shape='note', style='filled', color='lightyellow')
                    graph.edge(parent_id, p_id)
                
                for c in cases:
                    c_label = c.get("description_cn", c["description"])[:50] + "..." if st.session_state.language == 'Chinese' else c["description"][:50] + "..."
                    c_id = c["id"]
                    related_p_summary = c.get("related_principle_summary", "")
                    
                    matched_p_id = None
                    for p in principles:
                        if p["summary"] == related_p_summary:
                            matched_p_id = p["id"]
                            break
                    
                    graph.node(c_id, c_label, shape='component', style='filled', color='lightgreen')
                    
                    if matched_p_id:
                        graph.edge(matched_p_id, c_id)
                
                st.graphviz_chart(graph)
            else:
                st.info(t['no_principles'])

# --- CONSULTANT PAGE ---
elif page == t['nav_consultant']:
    st.title(t['consultant_title'])
    st.markdown("### Describe your situation")
    
    user_input = st.text_area(t['input_placeholder'], height=150)
    
    if st.button(t['get_advice'], type="primary"):
        if user_input:
            with st.spinner(t['processing']):
                try:
                    # Pass language to generate_advice
                    result = st.session_state.decision.generate_advice(user_input, language=st.session_state.language)
                    
                    advice = result["advice"]
                    related_principles = result["related_principles"]
                    
                    st.markdown(t['advice_title'])
                    st.markdown(advice)
                    
                    # Store context for saving
                    st.session_state.last_advice = advice
                    st.session_state.last_input = user_input
                    st.session_state.last_related_principles = related_principles
                    
                except Exception as e:
                    st.error(f"Error generating advice: {e}")
        else:
            st.warning("Please enter a problem description.")

    # Save User Case Feature
    if 'last_advice' in st.session_state:
        st.markdown("---")
        st.markdown(f"### 📥 {t['save_case']}")
        st.write("Was this a useful case for your future self? Save it!")
        
        if st.button(t['save_case']):
            related_principles = st.session_state.get("last_related_principles", [])
            
            # If no principles found, save as generic
            if not related_principles:
                related_principles = ["General"]
            
            # Save a case for EACH related principle (Splitting)
            saved_count = 0
            for p_summary in related_principles:
                # We need to find the principle summary string. 
                # The result from generate_advice returns summaries extracted from metadata.
                # If it's "Unknown", we might want to skip or label as General.
                
                new_case = {
                    "id": str(uuid.uuid4()),
                    "description": st.session_state.last_input,
                    "source": "User Input",
                    "type": "User Personal",
                    "advice_given": st.session_state.last_advice,
                    "related_principle_summary": p_summary
                }
                st.session_state.kb.save_case(new_case)
                saved_count += 1
            
            st.success(f"{t['save_success']} (Linked to {saved_count} principles)")
            del st.session_state.last_advice # Reset

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("v0.1 MVP")
