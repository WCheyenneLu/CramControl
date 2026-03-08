import json
import os
import sys
import tempfile
from pathlib import Path
import streamlit as st
from datetime import date
from optimizer import optimize_weekly_schedule

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from test_api import parse_syllabus_pdf

st.set_page_config(page_title="Cram Control Calendar", layout="wide")
st.title("Cram Control: Weekly Optimized Calendar")
st.write("Upload syllabus PDFs to automatically generate optimized weekly time allocations.")

# UI Settings
col1, col2 = st.columns(2)
with col1:
    quarter_start_input = st.date_input("Quarter start date", value=date.today())
    year_fallback_input = st.number_input("Year fallback", value=date.today().year, min_value=2000, max_value=2100)
with col2:
    light_week_max = st.number_input("Light week max hours", min_value=0.0, max_value=80.0, value=12.0, step=1.0)
    heavy_week_max = st.number_input("Heavy week max hours", min_value=0.0, max_value=100.0, value=25.0, step=1.0)

api_key = st.text_input("Gemini API Key", type="password")
uploaded_files = st.file_uploader("Upload syllabus PDF(s)", type=["pdf"], accept_multiple_files=True)

if st.button("Generate Master Calendar", type="primary"):
    if not api_key:
        st.error("Enter a Gemini API key.")
    elif not uploaded_files:
        st.error("Upload at least one PDF first.")
    else:
        all_combined_tasks = []
        raw_results = {}

        # 1. EXTRACT FROM ALL PDFS
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                # Get the raw tasks and ML estimates for this specific class
                data = parse_syllabus_pdf(tmp_path, api_key, quarter_start_input, year_fallback_input, light_week_max, heavy_week_max)
                raw_results[uploaded_file.name] = data
                
                course_label = data.get("course_code") or data.get("course_name") or uploaded_file.name
                
                # Tag each task with its course so we know where it came from in the master calendar
                for item in data.get("schedule_items", []):
                    item["parent_course"] = course_label
                    all_combined_tasks.append(item)
                    
            except Exception as exc:
                st.error(f"Extraction failed for {uploaded_file.name}: {exc}")
            finally:
                try: 
                    os.remove(tmp_path)
                except OSError: 
                    pass

        # 2. RUN GLOBAL OPTIMIZATION
        if all_combined_tasks:
            st.success(f"Successfully extracted {len(all_combined_tasks)} total tasks across {len(uploaded_files)} courses. Running Global Optimizer...")
            
            # We wrap the optimizer in a try-except just in case the math solver fails
            try:
                master_weekly_schedule = optimize_weekly_schedule(all_combined_tasks, quarter_start_input)
                
                # 3. RENDER THE MASTER UI
                st.write("### 📅 Your Master Optimized Schedule")
                
                # SAFETY CHECK: Make sure we actually have tabs to draw
                tab_labels = list(master_weekly_schedule.keys())
                if not tab_labels:
                    st.warning("No weekly schedule data was generated!")
                else:
                    tabs = st.tabs(tab_labels)
                    
                    for idx, (week_name, week_data) in enumerate(master_weekly_schedule.items()):
                        with tabs[idx]:
                            tasks = week_data.get("tasks", [])
                            alloc = week_data.get("optimal_allocation", {})
                            
                            if not tasks:
                                st.info("No tasks scheduled for this week! Take a break.")
                                continue
                                
                            st.caption(f"Status: {week_data.get('optimization_status')}")
                            
                            if alloc:
                                st.markdown(f"**Life Blocks:** 🧘‍♂️ Self Care: {alloc.get('general_life', {}).get('self_care', 0)} hrs | 🍻 Social: {alloc.get('general_life', {}).get('social', 0)} hrs")
                            
                            st.markdown("---")
                            
                            # Group tasks by Course for cleaner UI
                            for task in tasks:
                                opt_hrs = task.get("optimal_hours", 0)
                                
                                with st.container(border=True):
                                    st.markdown(f"#### {task.get('assignment_name')}")
                                    st.markdown(f"🎓 **Course:** {task.get('parent_course')} | ⏱️ **Optimal Time:** `{opt_hrs} hours`")
                                    
                                    # Show if the date failed parsing and went to 'Unscheduled'
                                    date_str = task.get('due_date_iso') or task.get('due_date')
                                    st.caption(f"Due: {date_str} | Weight: {task.get('weighting')} | Type: {task.get('task_type')}")
                    
                    with st.expander("View Raw JSON Payload"):
                        st.json(raw_results)

                # Move the download button inside the successful run block
                st.download_button(
                    label="Download Full Calendar JSON", 
                    data=json.dumps(master_weekly_schedule, indent=2), 
                    file_name="master_optimized_calendar.json", 
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Optimization failed: {e}")