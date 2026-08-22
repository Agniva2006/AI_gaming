import os
import multiprocessing
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_backend():
    import uvicorn
    # Import locally to avoid pygame initializing in backend process
    from backend.app import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

def main():
    # Start backend server in a separate process
    backend_process = multiprocessing.Process(target=run_backend, daemon=True)
    backend_process.start()
    
    # Start pygame frontend
    from engine.game import Game
    game = Game()
    game.run()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
