import streamlit as st

st.set_page_config(page_title="About - Risk Engine Pro", page_icon="ℹ️")

st.title("ℹ️ About Risk Engine Pro")

st.markdown("""
### Methodology
This simulator uses a **Monte Carlo Engine** to project portfolio outcomes based on 15 years of historical market data.

* **Fat-Tail Risk:** Unlike standard models that use a Normal Distribution, we use a **Student-T Distribution** ($df=3$) to better capture market crashes.
* **Sequence of Returns (SOR):** The 'Stress Test' toggle forces the worst 10% of historical outcomes to occur at the start of the timeline, simulating the impact of a market crash immediately after retirement.
* **Inflation Adjustment:** Returns can be toggled to 'Real' terms to show purchasing power rather than just nominal dollar amounts.

### Disclaimer
*Past performance is not indicative of future results. This tool is for educational purposes only.*
""")

if st.button("← Back to Simulator"):
    st.switch_page("risk_engine.py")