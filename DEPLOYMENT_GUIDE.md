# 🌐 Deployment Guide - Distracted Driver Detection

> 🚀 **Live Production URL:** **[https://distracted-driver-manish.streamlit.app/](https://distracted-driver-manish.streamlit.app/)**  
> Currently deployed and active on Streamlit Community Cloud.

---

## 🚀 Option 1: Streamlit Community Cloud (Recommended - 100% Free)

Streamlit Cloud provides free cloud hosting with a permanent public URL (e.g., `https://your-project.streamlit.app`).

### Prerequisites:
- A free GitHub account ([github.com](https://github.com))
- A free Streamlit Cloud account ([share.streamlit.io](https://share.streamlit.io))

### Steps:
1. **Push your project to GitHub**:
   In your project directory terminal:
   ```bash
   git add .
   git commit -m "Complete Distracted Driver Detection project"
   git push origin master
   ```
   *(Note: The `.gitignore` file already excludes the 2GB raw dataset, so only the code and ~50MB models will be uploaded, staying well within GitHub limits).*

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
   - Click **"Create app"** / **"New app"**.
   - Fill in the details:
     - **Repository:** `YourUsername/Distracted-Driver-Detection`
     - **Branch:** `master` (or `main`)
     - **Main file path:** `app.py`
     - **App URL:** (Choose a custom name, e.g. `driver-safety-ai`)
   - Click **Deploy!**

3. **Done!** Within 2-3 minutes, your live URL will be ready to share with professors and examiners.

---

## 📱 Option 2: Live Local Network Sharing (Instant - No Internet Needed)

If you are presenting inside a classroom or lab and want your professor to test the model directly from their own phone or laptop:

1. Connect your laptop and your professor's device to the **same Wi-Fi** (or your mobile hotspot).
2. Run your app:
   ```bash
   python run_app.py
   ```
3. Terminal will output two URLs:
   ```
   Local URL:   http://localhost:8501
   Network URL: http://192.168.1.XX:8501
   ```
4. Ask your professor or friends to open the **Network URL** on their mobile browser. They can immediately upload photos from their phone and test the model live!

---

## 🔗 Option 3: Instant Public Tunnel via Ngrok (No Git Push Required)

If you want an instant HTTPS link that works worldwide without uploading to GitHub:

1. Install ngrok:
   ```bash
   pip install pyngrok
   ```
2. Start the tunnel in a separate terminal:
   ```bash
   ngrok http 8501
   ```
3. Copy the generated `Forwarding` URL (e.g. `https://abc-123.ngrok-free.app`) and share it with anyone!

---

## 🐳 Option 4: Docker Container Deployment

A production-ready `Dockerfile` is included in the project root:

```bash
# Build the Docker image
docker build -t distracted-driver-detection .

# Run the container locally
docker run -p 8501:8501 distracted-driver-detection
```
This image can be deployed directly to:
- **Render.com** (Free Web Service)
- **Railway.app**
- **Hugging Face Spaces** (Docker SDK)
- **Google Cloud Run / AWS ECS**
