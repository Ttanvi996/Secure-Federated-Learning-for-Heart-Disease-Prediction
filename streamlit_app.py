import streamlit as st
import pandas as pd
import os
import subprocess
import time
import threading
import matplotlib.pyplot as plt

st.set_page_config(page_title="Federated Learning Dashboard", layout="wide")

# --- Utilities ---
def extract_latest_accuracy():
    if os.path.exists("training_log.txt"):
        with open("training_log.txt", "r") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Test Accuracy:" in line:
                    try:
                        return float(line.strip().split(":")[1].replace("%", "").strip())
                    except:
                        return None
    return None

def extract_time_metrics():
    aggregation_times = []
    decryption_times = []
    if os.path.exists("training_log.txt"):
        with open("training_log.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Aggregation time:" in line:
                    try:
                        t = float(line.strip().split(":")[1].replace("seconds", "").strip())
                        aggregation_times.append(t)
                    except:
                        pass
                if "Decryption time for client:" in line:
                    try:
                        t = float(line.strip().split(":")[1].replace("seconds", "").strip())
                        decryption_times.append(t)
                    except:
                        pass
    return aggregation_times, decryption_times

def show_final_accuracy():
    accuracy = extract_latest_accuracy()
    if accuracy is not None:
        st.metric(label="🎯 Final Global Accuracy After 3 Rounds", value=f"{accuracy:.2f}%")
    else:
        st.warning("⚠️ No accuracy yet. Please train the model.")

def show_time_summary(aggregation_times, decryption_times):
    if aggregation_times and decryption_times:
        avg_agg = sum(aggregation_times) / len(aggregation_times)
        avg_decrypt = sum(decryption_times) / len(decryption_times)

        cols = st.columns(2)
        cols[0].metric(label="⚡ Avg Aggregation Time", value=f"{avg_agg:.4f} sec")
        cols[1].metric(label="🔒 Avg Decryption Time", value=f"{avg_decrypt:.4f} sec")

def plot_time_graph(aggregation_times, decryption_times):
    if aggregation_times and decryption_times:
        rounds = list(range(1, len(aggregation_times)+1))
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, aggregation_times, marker='o', label='Aggregation Time')
        plt.plot(rounds, decryption_times[:len(rounds)], marker='x', linestyle='--', label='Decryption Time (avg per round)')
        plt.xlabel("Round")
        plt.ylabel("Time (seconds)")
        plt.title("⏱️ Aggregation vs Decryption Time per Round")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

# --- Session Initialization ---
if "training_complete" not in st.session_state:
    st.session_state["training_complete"] = False

# --- Title ---
st.title("🧠 Federated Learning for Heart Disease Prediction")

# --- Accuracy Section (Dynamic based on session state) ---
if st.session_state["training_complete"]:
    show_final_accuracy()
    aggregation_times, decryption_times = extract_time_metrics()
    show_time_summary(aggregation_times, decryption_times)
else:
    st.info("🚀 Start training to view the final accuracy and time metrics here.")

# --- Show client datasets ---
st.header("📊 Sample Client Data")
cols = st.columns(3)
for i in range(1, 4):
    try:
        df = pd.read_csv(f"datasets/client_{i}_data.csv")
        with cols[i - 1]:
            st.write(f"Client {i}")
            st.dataframe(df.head())
    except FileNotFoundError:
        st.warning(f"Missing: client_{i}_data.csv")

# --- Training Trigger ---
st.header("🔁 Federated Learning Process")
if st.button("Start Federated Training (Coordinator + Clients)"):
    with st.spinner("Running Federated Learning System..."):

        coordinator_path = os.path.abspath("coordinator.py")
        coordinator_process = subprocess.Popen(
            ["python3", coordinator_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        st.write("🧠 Coordinator started.")
        time.sleep(5)  # wait for server

        for round_num in range(1, 4):
            st.markdown(f"### 🔄 Round {round_num}")
            threads = []

            for client_id in range(1, 4):
                client_path = os.path.abspath(f"client-{client_id}.py")
                st.write(f"🚀 Launching Client {client_id}")

                def run_client(path=client_path, cid=client_id):
                    result = subprocess.run(["python3", path], capture_output=True, text=True)
                    if result.returncode != 0:
                        st.error(f"❌ Client {cid} failed:\n\n```{result.stderr}\n```")
                    else:
                        st.success(f"✅ Client {cid} completed.")
                        st.code(result.stdout)

                thread = threading.Thread(target=run_client)
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

            time.sleep(2)

        try:
            stdout, stderr = coordinator_process.communicate(timeout=30)
            st.success("✅ Coordinator finished.")
            if stdout:
                st.code(stdout)
            if stderr:
                st.error(f"Coordinator stderr:\n\n```{stderr}\n```")
        except subprocess.TimeoutExpired:
            coordinator_process.kill()
            st.error("❌ Coordinator timed out.")

        # ✅ After training complete, set flag
        st.session_state["training_complete"] = True
        st.rerun()

# --- Show Logs ---
if os.path.exists("training_log.txt"):
    with open("training_log.txt", "r") as f:
        logs = f.read()
    st.header("📈 Training Logs")
    st.text_area("Logs", logs, height=400)

    # Also show time graph
    st.header("⏳ Aggregation and Decryption Time per Round")
    aggregation_times, decryption_times = extract_time_metrics()
    plot_time_graph(aggregation_times, decryption_times)
