#!/usr/bin/env python3
"""
Batch Face Capture Script
Capture both boss and others images in one session
"""

import os
import sys
import subprocess
import time

def run_capture_session(category, count, auto=False):
    """Run a capture session for specified category"""
    cmd = [sys.executable, "capture_and_crop.py", category, "-n", str(count)]
    if auto:
        cmd.append("--auto")
    
    print(f"\n🚀 Starting {category} capture session...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {category} capture: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹ {category} capture interrupted by user")
        return False

def main():
    print("🎯 Batch Face Capture for BossSensor Training")
    print("=" * 50)
    print("This script helps you capture both boss and others images efficiently")
    print()
    
    # Get user preferences
    print("📋 Setup Questions:")
    
    try:
        boss_count = int(input("How many BOSS images to capture? (default 25): ") or "25")
        others_count = int(input("How many OTHERS images to capture? (default 25): ") or "25")
    except ValueError:
        print("❌ Invalid input. Using defaults.")
        boss_count = 25
        others_count = 25
    
    auto_mode = input("Use automatic capture mode? (y/N): ").lower().startswith('y')
    
    print(f"\n📊 Capture Plan:")
    print(f"   • Boss images: {boss_count}")
    print(f"   • Others images: {others_count}")
    print(f"   • Mode: {'Automatic' if auto_mode else 'Manual'}")
    print(f"   • Total session time: ~{(boss_count + others_count) // 10 + 5} minutes")
    
    input("\nPress Enter to start capturing...")
    
    # Session 1: Capture boss images
    print("\n" + "="*60)
    print("🎯 SESSION 1: BOSS IMAGES")
    print("="*60)
    print("Position yourself (the boss) in front of the camera.")
    print("Make sure you have good lighting and clear face visibility.")
    input("Press Enter when ready...")
    
    success1 = run_capture_session("boss", boss_count, auto_mode)
    
    if not success1:
        print("❌ Boss capture failed. Exiting.")
        return
    
    # Session 2: Capture others images
    print("\n" + "="*60)
    print("👥 SESSION 2: OTHERS IMAGES")  
    print("="*60)
    print("Now capture other people (colleagues, family, friends).")
    print("These should NOT be the boss - anyone else is fine.")
    print("You can have multiple people, or the same person from different angles.")
    input("Press Enter when ready...")
    
    success2 = run_capture_session("others", others_count, auto_mode)
    
    # Final summary
    print("\n" + "="*60)
    print("🎊 BATCH CAPTURE COMPLETE!")
    print("="*60)
    
    if success1 and success2:
        print("✅ Both sessions completed successfully!")
        print("\n🎯 Ready for training! Run:")
        print("   python train_model.py")
        print("\n💡 Training Tips:")
        print("   • Review captured images in faces/boss/ and faces/others/")
        print("   • Delete any poor quality images before training")
        print("   • More variety in others images = better accuracy")
    else:
        print("⚠ Some sessions had issues. Check the results and re-run if needed.")
    
    # Show final counts
    try:
        boss_files = len([f for f in os.listdir("faces/boss") if f.endswith('.jpg')])
        others_files = len([f for f in os.listdir("faces/others") if f.endswith('.jpg')])
        
        print(f"\n📊 Final Training Data:")
        print(f"   • Boss images: {boss_files}")
        print(f"   • Others images: {others_files}")
        print(f"   • Total: {boss_files + others_files}")
        
    except FileNotFoundError:
        print("⚠ Could not count final images - directories may not exist")

if __name__ == "__main__":
    main()
