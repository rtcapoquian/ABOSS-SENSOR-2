#!/usr/bin/env python3
"""
Test Screen Overlay
Simple test to verify the overlay functionality works
"""

import tkinter as tk
from PIL import Image, ImageTk
import os
import time

def test_overlay():
    print("🧪 Testing screen overlay...")
    
    # Create tkinter overlay window
    overlay_root = tk.Tk()
    overlay_root.title("Test Overlay")
    
    # Make window fullscreen and always on top
    overlay_root.attributes('-fullscreen', True)
    overlay_root.attributes('-topmost', True)
    overlay_root.attributes('-alpha', 1.0)  # Fully opaque
    overlay_root.configure(bg='blue')
    
    # Get screen dimensions
    screen_width = overlay_root.winfo_screenwidth()
    screen_height = overlay_root.winfo_screenheight()
    
    print(f"📺 Screen size: {screen_width}x{screen_height}")
    
    # Check if working screen exists
    if os.path.exists("working_screen.png"):
        try:
            # Load and resize image
            pil_image = Image.open("working_screen.png")
            pil_image = pil_image.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Create label to display image
            image_label = tk.Label(overlay_root, image=tk_image, bg='black')
            image_label.pack(fill=tk.BOTH, expand=True)
            image_label.image = tk_image
            
            print("✅ Working screen image loaded")
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            # Fallback
            fallback_label = tk.Label(overlay_root, text="TEST OVERLAY\nPress ESC to close", 
                                    font=("Arial", 48), fg="white", bg="blue")
            fallback_label.pack(expand=True)
    else:
        # No image, show text
        test_label = tk.Label(overlay_root, text="TEST OVERLAY\nworking_screen.png not found\nPress ESC to close", 
                            font=("Arial", 32), fg="white", bg="blue")
        test_label.pack(expand=True)
        print("⚠ working_screen.png not found, showing test text")
    
    # Close function
    def close_overlay():
        overlay_root.destroy()
        print("🔚 Test overlay closed")
    
    # Auto-close after 5 seconds
    overlay_root.after(5000, close_overlay)
    
    # Bind escape key
    overlay_root.bind('<Escape>', lambda e: close_overlay())
    
    print("🎯 Overlay active for 5 seconds (or press ESC)")
    print("   This should cover your entire screen!")
    
    # Run overlay
    overlay_root.mainloop()
    
    print("✅ Test completed!")

if __name__ == "__main__":
    test_overlay()
