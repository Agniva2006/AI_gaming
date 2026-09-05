import argparse
import sys
import time
from rl_env.behavioral_cloning import generate_expert_dataset, BCTrainer

def main():
    parser = argparse.ArgumentParser(description="Offline Behavioral Cloning Pre-Trainer for StadiumAI")
    parser.add_argument("--episodes", type=int, default=12, help="Number of expert simulation episodes to collect")
    parser.add_argument("--max-steps", type=int, default=250, help="Max steps per episode")
    parser.add_argument("--epochs", type=int, default=5, help="Supervised training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for gradient updates")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    print("=" * 60)
    print("[AI] STADIUM AI: OFFLINE BEHAVIORAL CLONING (IMITATION LEARNING)")
    print("=" * 60)
    print(f"Parameters: {args.episodes} episodes, {args.epochs} epochs, batch size {args.batch_size}, lr {args.lr}")

    start_time = time.time()
    
    # 1. Collect demonstration dataset
    obs, acts, vals = generate_expert_dataset(num_episodes=args.episodes, max_steps=args.max_steps)

    # 2. Supervised Training
    trainer = BCTrainer()
    results = trainer.train(
        obs, acts, vals,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"[SUCCESS] Behavioral Cloning Pre-Training Complete in {elapsed:.1f}s!")
    print(f"Total Demonstrations: {results['samples']}")
    print(f"Final Prediction Accuracy: {results['accuracy']}%")
    print(f"Final Supervised Loss: {results['final_loss']}")
    print("The RL policy is now pre-initialized with expert passing, shooting, and positional skills.")
    print("=" * 60)

if __name__ == "__main__":
    main()
