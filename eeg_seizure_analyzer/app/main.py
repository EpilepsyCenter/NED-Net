"""Streamlit app entry point.

Run with: streamlit run eeg_seizure_analyzer/app/main.py
"""

import streamlit as st

st.set_page_config(
    page_title="EEG Seizure Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    ("Upload", "📂"),
    ("Viewer", "📈"),
    ("Seizures", "⚡"),
    ("Spikes", "📌"),
    ("Validation", "✅"),
    ("Export", "💾"),
]


def _set_page(page_name: str):
    st.session_state["current_page"] = page_name


def main():
    # Seed all widget defaults into session state (once)
    from eeg_seizure_analyzer.app.components import init_session_defaults
    init_session_defaults()

    # Initialize default page
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Upload"

    current_page = st.session_state["current_page"]

    # ── Top navigation bar ────────────────────────────────────────────
    cols = st.columns(len(PAGES))
    for col, (name, icon) in zip(cols, PAGES):
        is_active = (name == current_page)
        btn_type = "primary" if is_active else "secondary"
        col.button(
            f"{icon} {name}",
            key=f"nav_{name}",
            type=btn_type,
            use_container_width=True,
            on_click=_set_page,
            args=(name,),
        )

    st.divider()

    # ── Page rendering ────────────────────────────────────────────────
    if current_page == "Upload":
        from eeg_seizure_analyzer.app.page_upload import render
        render()
    elif current_page == "Viewer":
        from eeg_seizure_analyzer.app.page_viewer import render
        render()
    elif current_page == "Seizures":
        from eeg_seizure_analyzer.app.page_seizures import render
        render()
    elif current_page == "Spikes":
        from eeg_seizure_analyzer.app.page_spikes import render
        render()
    elif current_page == "Validation":
        from eeg_seizure_analyzer.app.page_validation import render
        render()
    elif current_page == "Export":
        from eeg_seizure_analyzer.app.page_export import render
        render()


if __name__ == "__main__":
    main()
