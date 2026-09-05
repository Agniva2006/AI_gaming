import os
import sys
import multiprocessing

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_backend():
    import uvicorn
    from backend.app import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

def main():
    print("=" * 60)
    print("[STADIUM AI] 2D FOOTBALL (HUMAN VS LEARNING RL AI)")
    print("=" * 60)
    print("[BACKEND] Starting FastAPI & WebSocket Gateway on port 8000...")
    
    backend_process = multiprocessing.Process(target=run_backend, daemon=True)
    backend_process.start()

    print("[DASHBOARD] Web Dashboard is live at: http://127.0.0.1:8000/dashboard")
    print("[ENGINE] Launching Pygame 2D Football Engine...")
    print("=" * 60)

    from engine.game import Game
    game = Game()
    game.run()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
