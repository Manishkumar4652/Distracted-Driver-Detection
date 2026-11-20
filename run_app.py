"""
Convenient launcher for the Distracted Driver Detection Dashboard.
Usage:
    python run_app.py
"""

import sys
import os
import subprocess

def setup_streamlit_config():
    """Silently configure Streamlit so it never prompts for email on first launch."""
    try:
        user_streamlit = os.path.expanduser("~/.streamlit")
        os.makedirs(user_streamlit, exist_ok=True)
        cred_path = os.path.join(user_streamlit, "credentials.toml")
        if not os.path.exists(cred_path):
            with open(cred_path, "w", encoding="utf-8") as f:
                f.write('[general]\nemail = ""\n')
        
        cfg_path = os.path.join(user_streamlit, "config.toml")
        if not os.path.exists(cfg_path):
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write('[browser]\ngatherUsageStats = false\n[server]\nheadless = false\n')
    except Exception:
        pass

def main():
    setup_streamlit_config()
    
    print("=" * 60)
    print("   🚗 DISTRACTED DRIVER DETECTION SYSTEM")
    print("   College Project Dashboard Launcher")
    print("=" * 60)
    print("\n[INFO] Starting Streamlit web application...")
    print("[INFO] If your browser does not open automatically, visit:")
    print("       👉 http://localhost:8501\n")
    
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--browser.gatherUsageStats", "false",
        "--server.headless", "false"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[INFO] Application stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n[ERROR] Failed to start application: {e}")

if __name__ == "__main__":
    main()
