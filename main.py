import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from groq import Groq

# === Page Setup ===
st.set_page_config(page_title="EV Range Predictor Dashboard", page_icon="⚡", layout="wide")

# === Load Model ===
@st.cache_resource
def load_model():
    return joblib.load("ev_range_predictor_reduced.pkl")

model = load_model()

# === Groq Client ===
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# === Sidebar ===
st.sidebar.header("📥 Input Parameters")

speed = st.sidebar.slider("🚙 Speed (Km/h)", 0.0, 200.0, 60.0)
acceleration = st.sidebar.slider("🏁 Acceleration (m/s²)", 0.0, 10.0, 1.5)
braking = st.sidebar.slider("🛑 Braking (m/s²)", 0.0, 10.0, 0.8)
prev_soc = st.sidebar.slider("⏪ Previous SoC (%)", 0.0, 100.0, 85.0)
temperature = st.sidebar.slider("🌡️ Temperature (°C)", -20.0, 60.0, 25.0)
terrain = st.sidebar.selectbox("🗻 Terrain", ["Flat", "Hilly"])
weather = st.sidebar.selectbox("🌦️ Weather", ["Normal", "Hot", "Cold", "Rainy"])
battery_health = st.sidebar.slider("🔧 Battery Health (%)", 50.0, 100.0, 100.0)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh every 30 seconds", value=False)
show_download = st.sidebar.checkbox("⬇️ Show CSV Download", value=True)

if auto_refresh:
    st_autorefresh(interval=30 * 1000, key="autorefresh")

# === Session State ===
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# === Helper Functions ===
def energy_rate(speed, terrain, weather, temperature):
    rate = 0.15
    if speed <= 50:
        rate = 0.12
    elif speed > 80:
        rate = 0.18
    if terrain == "Hilly":
        rate *= 1.2
    if weather in ["Hot", "Cold", "Rainy"]:
        rate *= 1.1
    if temperature < 0:
        rate *= 1.25
    elif temperature < 10:
        rate *= 1.15
    elif temperature > 40:
        rate *= 1.10
    elif 20 <= temperature <= 30:
        rate *= 0.97
    return rate

def efficiency_score(speed, acc, brake, terrain, weather, temperature):
    score = 100
    if speed > 100: score -= 15
    elif speed > 80: score -= 10
    if speed < 30: score -= 5
    if acc > 3: score -= 10
    if brake > 3: score -= 10
    if terrain == "Hilly": score -= 5
    if weather in ["Hot", "Cold", "Rainy"]: score -= 5
    if temperature < 0: score -= 15
    elif temperature < 10: score -= 8
    elif temperature > 40: score -= 8
    elif 20 <= temperature <= 30: score += 3
    return max(min(score, 100), 0)

def get_ai_analysis(context):
    prompt = f"""You are an expert EV driving coach and battery analyst.

Current driving conditions and prediction results:
- Speed: {context['speed']} Km/h
- Acceleration: {context['acceleration']} m/s²
- Braking: {context['braking']} m/s²
- Terrain: {context['terrain']}
- Weather: {context['weather']}
- Temperature: {context['temperature']}°C
- Battery Health: {context['battery_health']}%
- Previous SoC: {context['prev_soc']}%
- Predicted SoC: {context['predicted_soc']:.2f}%
- Estimated Range: {context['predicted_range']:.2f} Km
- SoC Drop: {context['soc_drop']:.2f}%
- Efficiency Score: {context['eff_score']}%

Please provide:
1. 🔍 A natural language summary of the driving situation
2. ⚠️ Key risks or concerns
3. 💡 3-5 specific actionable tips to improve range
4. 🔋 Battery health advice if relevant
5. ✅ Overall assessment (Excellent/Good/Fair/Poor) with reason

Be concise, friendly and practical."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def get_whatif_advice(context):
    prompt = f"""You are an EV range optimization expert.

Current conditions:
- Speed: {context['speed']} Km/h
- Terrain: {context['terrain']}
- Weather: {context['weather']}
- Temperature: {context['temperature']}°C
- Battery Health: {context['battery_health']}%
- Predicted Range: {context['predicted_range']:.2f} Km
- Efficiency Score: {context['eff_score']}%

Give exactly 3 specific What-If scenarios showing how changing one parameter improves range.
Format: "If you [change], your range could improve by approximately [X]%"
Be specific with numbers. Keep it brief."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def get_chat_response(user_message, context):
    system_prompt = f"""You are an expert EV driving coach embedded in an EV Range Prediction Dashboard.

Current driving context:
- Speed: {context['speed']} Km/h, Acceleration: {context['acceleration']} m/s², Braking: {context['braking']} m/s²
- Terrain: {context['terrain']}, Weather: {context['weather']}, Temperature: {context['temperature']}°C
- Battery Health: {context['battery_health']}%, Previous SoC: {context['prev_soc']}%
- Predicted SoC: {context['predicted_soc']:.2f}%, Estimated Range: {context['predicted_range']:.2f} Km
- SoC Drop: {context['soc_drop']:.2f}%, Efficiency Score: {context['eff_score']}%

Answer questions about EV range, battery, or driving. Be concise, helpful and friendly."""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.chat_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        messages=messages
    )
    return response.choices[0].message.content

# === Main Title ===
st.title("⚡ Smart EV Range Prediction Dashboard")
st.markdown("Predict your **State of Charge** and **Range** with AI-powered insights.")
st.markdown("---")

# === Predict Button ===
if st.button("🔮 Predict Range", type="primary"):

    input_data = pd.DataFrame([{
        "Speed (Km/h)": speed,
        "Acceleration (m/s²)": acceleration,
        "Braking (m/s²)": braking,
        "Prev_SoC": prev_soc,
        "Terrain": terrain,
        "Weather": weather
    }])

    predicted_soc = model.predict(input_data)[0]

    battery_capacity = 40 * (battery_health / 100)
    consumption_rate = energy_rate(speed, terrain, weather, temperature)
    remaining_energy = (predicted_soc / 100) * battery_capacity
    predicted_range = remaining_energy / consumption_rate
    soc_drop = prev_soc - predicted_soc
    eff_score = efficiency_score(speed, acceleration, braking, terrain, weather, temperature)

    context = {
        "speed": speed, "acceleration": acceleration, "braking": braking,
        "prev_soc": prev_soc, "temperature": temperature, "terrain": terrain,
        "weather": weather, "battery_health": battery_health,
        "predicted_soc": predicted_soc, "predicted_range": predicted_range,
        "soc_drop": soc_drop, "eff_score": eff_score
    }
    st.session_state.last_context = context
    st.session_state.history.append({
        "Speed": speed, "Terrain": terrain, "Weather": weather,
        "Temp (°C)": temperature, "Battery Health (%)": battery_health,
        "Predicted SoC (%)": round(predicted_soc, 2),
        "Range (Km)": round(predicted_range, 2),
        "Efficiency (%)": eff_score
    })

    # === Metrics ===
    st.header("📊 Prediction Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔋 Predicted SoC", f"{predicted_soc:.2f}%", f"{-soc_drop:.2f}%")
    c2.metric("📏 Estimated Range", f"{predicted_range:.2f} Km")
    c3.metric("⬇️ SoC Drop", f"{soc_drop:.2f}%")
    c4.metric("⚙️ Efficiency", f"{eff_score}%")
    c5.metric("🔧 Usable Battery", f"{battery_capacity:.1f} kWh")

    if predicted_range < 50:
        st.error(f"⚠️ Low Range Warning! Only {predicted_range:.1f} Km remaining. Charge soon!")
    elif predicted_range < 100:
        st.warning(f"🟡 Moderate range ({predicted_range:.1f} Km). Consider charging soon.")

    st.markdown("---")

    # =====================
    # === IMPROVED CHARTS ===
    # =====================

    st.subheader("📊 Visual Insights")
    st.caption("These charts help you understand your battery and range at a glance.")

    # --- Chart 1: SoC Gauge ---
    st.markdown("#### 🔋 Battery Charge Level")
    st.caption("Shows how much battery charge is remaining. Green = Safe, Orange = Low, Red = Critical.")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_soc,
        delta={
            'reference': prev_soc,
            'increasing': {'color': "green"},
            'decreasing': {'color': "red"}
        },
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "gray"},
            'bar': {'color': "#00CC96" if predicted_soc > 50 else "#FFA500" if predicted_soc > 20 else "#FF4444"},
            'steps': [
                {'range': [0, 20], 'color': "#FFE5E5", 'name': 'Critical'},
                {'range': [20, 50], 'color': "#FFF3E0", 'name': 'Low'},
                {'range': [50, 80], 'color': "#FFFDE7", 'name': 'Moderate'},
                {'range': [80, 100], 'color': "#E8F5E9", 'name': 'Safe'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 20
            }
        }
    ))
    fig_gauge.update_layout(
        height=300,
        margin=dict(t=40, b=10, l=30, r=30),
        annotations=[
            dict(x=0.18, y=0.18, text="🔴 Critical", showarrow=False, font=dict(size=11, color="red")),
            dict(x=0.38, y=0.05, text="🟠 Low", showarrow=False, font=dict(size=11, color="orange")),
            dict(x=0.62, y=0.05, text="🟡 Moderate", showarrow=False, font=dict(size=11, color="goldenrod")),
            dict(x=0.82, y=0.18, text="🟢 Safe", showarrow=False, font=dict(size=11, color="green")),
        ]
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # --- Chart 2: Battery Used vs Remaining ---
    st.markdown("#### 🔋 How Much Battery Is Left?")
    st.caption("Shows exactly how many kWh are used vs remaining in your battery.")

    used_energy = battery_capacity - remaining_energy
    fig_battery = go.Figure()
    fig_battery.add_trace(go.Bar(
        y=['Battery'],
        x=[used_energy],
        name='Used 🔴',
        orientation='h',
        marker=dict(color='#FF6B6B', line=dict(color='#CC0000', width=1)),
        text=f"{used_energy:.1f} kWh used",
        textposition='inside',
        insidetextanchor='middle'
    ))
    fig_battery.add_trace(go.Bar(
        y=['Battery'],
        x=[remaining_energy],
        name='Remaining 🟢',
        orientation='h',
        marker=dict(color='#51CF66', line=dict(color='#2E7D32', width=1)),
        text=f"{remaining_energy:.1f} kWh left",
        textposition='inside',
        insidetextanchor='middle'
    ))
    fig_battery.update_layout(
        barmode='stack',
        height=180,
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(title="Battery Capacity (kWh)", showgrid=True),
        yaxis=dict(showticklabels=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_battery, use_container_width=True)

    # --- Chart 3: Energy Consumption vs Speed ---
    st.markdown("#### 📈 How Speed Affects Your Range")
    st.caption("Higher speed = more energy used per km. The red line shows your current speed.")

    speeds = list(range(10, 141, 5))
    rates = [energy_rate(s, terrain, weather, temperature) for s in speeds]
    ranges_at_speed = [((predicted_soc / 100) * battery_capacity) / r for r in rates]

    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(
        x=speeds,
        y=ranges_at_speed,
        mode='lines',
        line=dict(color='#339AF0', width=3),
        fill='tozeroy',
        fillcolor='rgba(51,154,240,0.1)',
        name='Estimated Range'
    ))
    fig_speed.add_vline(
        x=speed,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"  ← You are here ({speed} Km/h)",
        annotation_position="top right",
        annotation_font=dict(color="red", size=12)
    )
    fig_speed.add_hrect(
        y0=0, y1=50,
        fillcolor="rgba(255,0,0,0.05)",
        line_width=0,
        annotation_text="⚠️ Danger Zone",
        annotation_position="top left",
        annotation_font=dict(color="red", size=11)
    )
    fig_speed.update_layout(
        height=350,
        xaxis=dict(title="Speed (Km/h)", showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(title="Estimated Range (Km)", showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        margin=dict(t=20, b=40, l=40, r=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    st.plotly_chart(fig_speed, use_container_width=True)

    # --- Chart 4: SoC Depletion Over Distance ---
    st.markdown("#### 📉 Battery Drain As You Drive")
    st.caption("Shows how your battery % drops as you travel. The red zone is where you should charge.")

    distances = list(range(0, int(predicted_range) + 15, 5))
    soc_over_distance = [
        max(predicted_soc - (consumption_rate * d / battery_capacity * 100), 0)
        for d in distances
    ]

    # Find where SoC hits 20%
    charge_distance = None
    for i, s in enumerate(soc_over_distance):
        if s <= 20:
            charge_distance = distances[i]
            break

    fig_depletion = go.Figure()
    fig_depletion.add_trace(go.Scatter(
        x=distances,
        y=soc_over_distance,
        mode='lines+markers',
        line=dict(color='#51CF66', width=3),
        marker=dict(size=5, color='#51CF66'),
        fill='tozeroy',
        fillcolor='rgba(81,207,102,0.1)',
        name='Battery Level'
    ))
    fig_depletion.add_hline(
        y=20,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="⚠️ Charge Now! (20%)",
        annotation_position="top right",
        annotation_font=dict(color="red", size=12)
    )
    if charge_distance:
        fig_depletion.add_vline(
            x=charge_distance,
            line_dash="dot",
            line_color="orange",
            line_width=2,
            annotation_text=f"  Charge by {charge_distance} Km",
            annotation_position="top right",
            annotation_font=dict(color="orange", size=11)
        )
    fig_depletion.add_hrect(
        y0=0, y1=20,
        fillcolor="rgba(255,0,0,0.07)",
        line_width=0
    )
    fig_depletion.update_layout(
        height=350,
        xaxis=dict(title="Distance Travelled (Km)", showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(title="Battery Level (%)", range=[0, 100], showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        margin=dict(t=20, b=40, l=40, r=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    st.plotly_chart(fig_depletion, use_container_width=True)

    # --- Chart 5: Temperature Impact ---
    st.markdown("#### 🌡️ How Temperature Affects Your Range")
    st.caption("Cold and hot temperatures reduce range. The green zone is the ideal temperature for best range.")

    temps = list(range(-20, 61, 2))
    ranges_at_temps = [
        ((predicted_soc / 100) * battery_capacity) / energy_rate(speed, terrain, weather, t)
        for t in temps
    ]

    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=temps,
        y=ranges_at_temps,
        mode='lines',
        line=dict(color='#FF922B', width=3),
        fill='tozeroy',
        fillcolor='rgba(255,146,43,0.1)',
        name='Range'
    ))
    # Highlight optimal zone
    fig_temp.add_vrect(
        x0=20, x1=30,
        fillcolor="rgba(0,200,0,0.1)",
        line_width=0,
        annotation_text="✅ Ideal Temp",
        annotation_position="top left",
        annotation_font=dict(color="green", size=11)
    )
    fig_temp.add_vline(
        x=temperature,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"  You: {temperature}°C",
        annotation_position="top right",
        annotation_font=dict(color="red", size=12)
    )
    fig_temp.update_layout(
        height=350,
        xaxis=dict(title="Temperature (°C)", showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(title="Estimated Range (Km)", showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        margin=dict(t=20, b=40, l=40, r=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # === Efficiency Score ===
    st.markdown("---")
    st.header("⚙️ Driving Efficiency Score")
    st.caption("A score out of 100 based on your speed, acceleration, braking, terrain, weather and temperature.")
    st.progress(eff_score / 100)
    if eff_score >= 80:
        st.success(f"✅ Excellent! Your driving efficiency is **{eff_score}%** — great conditions for maximum range!")
    elif eff_score >= 60:
        st.warning(f"⚠️ Moderate. Your driving efficiency is **{eff_score}%** — some room for improvement.")
    else:
        st.error(f"❌ Poor. Your driving efficiency is **{eff_score}%** — conditions are significantly reducing your range.")

    st.markdown("---")

    # === AI Analysis ===
    st.header("🤖 AI Driving Coach Analysis")
    with st.spinner("Analysing your driving conditions..."):
        ai_analysis = get_ai_analysis(context)
    st.markdown(ai_analysis)

    st.markdown("---")

    # === What-If Advisor ===
    st.header("🔀 What-If Range Advisor")
    with st.spinner("Generating what-if scenarios..."):
        whatif = get_whatif_advice(context)
    st.info(whatif)

    st.markdown("---")

    # === CSV Download ===
    if show_download:
        st.header("📂 Download Prediction")
        download_df = pd.DataFrame([{
            "Speed (Km/h)": speed,
            "Acceleration (m/s²)": acceleration,
            "Braking (m/s²)": braking,
            "Previous SoC (%)": prev_soc,
            "Temperature (°C)": temperature,
            "Terrain": terrain,
            "Weather": weather,
            "Battery Health (%)": battery_health,
            "Predicted SoC (%)": round(predicted_soc, 2),
            "Predicted Range (Km)": round(predicted_range, 2),
            "SoC Drop (%)": round(soc_drop, 2),
            "Driving Efficiency (%)": eff_score,
        }])
        st.download_button(
            label="⬇️ Download CSV",
            data=download_df.to_csv(index=False),
            file_name="ev_prediction_result.csv",
            mime="text/csv",
        )

else:
    st.info("🔹 Set your parameters in the sidebar and click **Predict Range** to get started.")

st.markdown("---")

# === Session History ===
if st.session_state.history:
    st.header("📋 Prediction History (This Session)")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

st.markdown("---")

# === AI Chat Assistant ===
st.header("💬 Ask Your EV AI Assistant")

if "last_context" not in st.session_state:
    st.info("👆 Run a prediction first to enable the AI chat assistant.")
else:
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about your EV range, battery, or driving tips..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = get_chat_response(prompt, st.session_state.last_context)
            st.markdown(reply)
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})

    if st.session_state.chat_messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_messages = []
            st.rerun()