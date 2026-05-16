import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from io import BytesIO
from datetime import datetime

# ================= PAGE =================
st.set_page_config(
    page_title="Heart AI Hospital System",
    page_icon="🏥",
    layout="wide"
)

# ================= LOAD =================
model = pickle.load(open("heart_disease_model.sav", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

X_test = pickle.load(open("X_test.pkl", "rb"))
Y_test = pickle.load(open("y_test.pkl", "rb"))

# ================= SESSION =================
if "result_ready" not in st.session_state:
    st.session_state.result_ready = False

st.markdown("# 🏥 Heart AI Hospital System")

# ================= SIDEBAR =================
view = st.sidebar.selectbox(
    "System Mode",
    ["Patient System 🧑‍⚕️", "Model Evaluation 📊"]
)

# =========================================================
# 🧑‍⚕️ PATIENT SYSTEM
# =========================================================
if view == "Patient System 🧑‍⚕️":

    st.subheader("🧾 Patient Examination")

    c1, c2 = st.columns(2)

    with c1:
        Chest_Pain = st.slider("Chest Pain", 0.0, 1.0, 0.0)
        Shortness_of_Breath = st.slider("Shortness of Breath", 0.0, 1.0, 0.0)
        Fatigue = st.slider("Fatigue", 0.0, 1.0, 0.0)
        Palpitations = st.slider("Palpitations", 0.0, 1.0, 0.0)
        Dizziness = st.slider("Dizziness", 0.0, 1.0, 0.0)
        Swelling = st.slider("Swelling", 0.0, 1.0, 0.0)
        Pain_Arms_Jaw_Back = st.slider("Pain Arms/Jaw/Back", 0.0, 1.0, 0.0)
        Cold_Sweats_Nausea = st.slider("Cold Sweats/Nausea", 0.0, 1.0, 0.0)
        High_BP = st.slider("High BP", 0.0, 1.0, 0.0)

    with c2:
        High_Cholesterol = st.slider("High Cholesterol", 0.0, 1.0, 0.0)
        Diabetes = st.slider("Diabetes", 0.0, 1.0, 0.0)
        Smoking = st.slider("Smoking", 0.0, 1.0, 0.0)
        Obesity = st.slider("Obesity", 0.0, 1.0, 0.0)
        Sedentary_Lifestyle = st.slider("Sedentary Lifestyle", 0.0, 1.0, 0.0)
        Family_History = st.slider("Family History", 0.0, 1.0, 0.0)
        Chronic_Stress = st.slider("Stress", 0.0, 1.0, 0.0)

        gender_label = st.selectbox("Gender", ["Male", "Female"])
        Gender = 0 if gender_label == "Male" else 1

        Age = st.number_input("Age", 1, 100, 25)

    # ================= ANALYZE =================
    if st.button("🔍 Analyze Patient"):

        input_data = np.array([[
            Chest_Pain, Shortness_of_Breath, Fatigue, Palpitations,
            Dizziness, Swelling, Pain_Arms_Jaw_Back, Cold_Sweats_Nausea,
            High_BP, High_Cholesterol, Diabetes, Smoking, Obesity,
            Sedentary_Lifestyle, Family_History, Chronic_Stress,
            Gender, Age
        ]])

        # 🔥 scaling
        input_data = scaler.transform(input_data)

        pred = model.predict(input_data)
        prob = model.predict_proba(input_data)[0][1]

        st.session_state.pred = pred
        st.session_state.prob = prob
        st.session_state.Age = Age
        st.session_state.gender_label = gender_label
        st.session_state.result_ready = True

        reasons = []
        if Chest_Pain > 0.5: reasons.append("Chest pain risk")
        if Smoking > 0.5: reasons.append("Smoking risk")
        if High_BP > 0.5: reasons.append("High blood pressure")
        if Diabetes > 0.5: reasons.append("Diabetes risk")
        if Obesity > 0.5: reasons.append("Obesity risk")

        st.session_state.reasons = reasons

    # ================= RESULT =================
    if st.session_state.result_ready:

        st.markdown("## 📊 Result")

        st.metric("Risk %", f"{st.session_state.prob*100:.2f}%")
        st.metric("Status", "RISK" if st.session_state.pred[0] == 1 else "SAFE")
        st.metric("Gender", st.session_state.gender_label)

        st.progress(float(st.session_state.prob))

        st.markdown("## 🧠 Explanation")

        if st.session_state.reasons:
            for r in st.session_state.reasons:
                st.write("•", r)
        else:
            st.success("No major risk detected")

        # ================= PDF FIXED =================
        def generate_pdf():
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            story.append(Paragraph("Heart AI Medical Report", styles["Title"]))
            story.append(Spacer(1, 10))

            story.append(Paragraph(f"Date: {datetime.now()}", styles["Normal"]))
            story.append(Paragraph(f"Age: {st.session_state.Age}", styles["Normal"]))
            story.append(Paragraph(f"Gender: {st.session_state.gender_label}", styles["Normal"]))
            story.append(Paragraph(f"Risk: {st.session_state.prob*100:.2f}%", styles["Normal"]))

            story.append(Spacer(1, 10))
            story.append(Paragraph("Clinical Review:", styles["Heading2"]))

            if st.session_state.reasons:
                for r in st.session_state.reasons:
                    story.append(Paragraph(r, styles["Normal"]))
            else:
                story.append(Paragraph("No risk detected", styles["Normal"]))

            doc.build(story)
            buffer.seek(0)
            return buffer

        if st.button("📄 Generate Report"):

            pdf_buffer = generate_pdf()

            st.download_button(
                label="📥 Download Report",
                data=pdf_buffer.getvalue(),
                file_name="Heart_Report.pdf",
                mime="application/pdf"
            )

    # ================= FEEDBACK + RATING =================
    st.markdown("---")
    st.subheader("💬 Feedback")

    rating = st.radio(
        "⭐ Rate the system:",
        ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
    )

    feedback = st.text_area("Write your feedback here")

    if st.button("Submit Feedback"):
        if feedback.strip() != "":
            st.success("Thank you for your feedback ❤️")
            st.info(f"Rating received: {rating}")
        else:
            st.warning("Please write feedback first")

# =========================================================
# 📊 MODEL EVALUATION
# =========================================================
else:

    y_pred = model.predict(X_test)

    acc = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)

    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    cm = confusion_matrix(Y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, y_prob)
    auc_score = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={auc_score:.2f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
st.markdown("🏥 AI Hospital System | Clinical Decision Support")