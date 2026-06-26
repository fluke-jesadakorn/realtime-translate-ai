import sys
import time
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("❌ Error: sounddevice library is not installed in your virtual environment.")
    print("Please make sure you are running this script inside the correct virtual environment:")
    print("  .venv/bin/python test_audio_input.py")
    sys.exit(1)

def main():
    print("=" * 60)
    print("🔊 AUDIO ROUTING & INPUT DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # 1. List input devices
    devices = sd.query_devices()
    input_devices = []
    blackhole_idx = None
    
    print("\n--- Available Input Devices ---")
    for i, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            name = dev["name"]
            channels = dev["max_input_channels"]
            sr = dev.get("default_samplerate", 0)
            print(f"[{i}] {name} ({channels} channels, default SR: {sr}Hz)")
            input_devices.append(i)
            if "blackhole" in name.lower():
                blackhole_idx = i

    # 2. Select device
    if len(input_devices) == 0:
        print("\n❌ Error: No input devices found on your system!")
        sys.exit(1)

    target_idx = blackhole_idx
    if target_idx is None:
        # Fallback to default input
        target_idx = sd.default.device[0]
        print(f"\n⚠️ BlackHole 2ch not found in names. Using default input device index: {target_idx}")
    else:
        print(f"\n✅ Found virtual device: [{target_idx}] {devices[target_idx]['name']}")

    # Prompt user or default to target_idx
    print(f"\nTargeting device index: [{target_idx}] {devices[target_idx]['name']}")
    print("Preparing to record for 5 seconds to test incoming audio signal...")
    print("⚠️ Please make sure you are playing test audio or speaking during this test! ⚠️")
    print("Starting in:")
    for count in range(3, 0, -1):
        print(f"  {count}...")
        time.sleep(1)
    
    print("\n🎤 Recording started... Play sound now!")
    duration = 5.0  # seconds
    sample_rate = int(devices[target_idx].get("default_samplerate", 44100))
    channels = min(2, devices[target_idx]["max_input_channels"])
    
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("✅ Recording completed.")
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        sys.exit(1)

    # 3. Analyze signal
    max_amp = float(np.abs(recording).max())
    rms = float(np.sqrt(np.mean(recording**2)))
    
    print("\n--- Analysis Results ---")
    print(f"Peak amplitude: {max_amp:.6f}")
    print(f"RMS (Average volume): {rms:.6f}")
    
    # 4. Diagnose issues
    print("\n--- Diagnosis & Recommended Action ---")
    if max_amp == 0.0:
        print("❌ CRITICAL: ABSOLUTE SILENCE DETECTED (Exactly 0.0)")
        print("This almost always means macOS is blocking audio capture.")
        print("\n💡 ACTION REQUIRED:")
        print("1. Go to System Settings (การตั้งค่าระบบ) -> Privacy & Security (ความเป็นส่วนตัวและความปลอดภัย) -> Microphone (ไมโครโฟน).")
        print("2. Ensure the terminal/application you are running this script in (e.g., Terminal, iTerm, VS Code, Cursor) is toggled ON (enabled).")
        print("3. If it is already ON, toggle it OFF, then toggle it back ON.")
        print("4. Restart your Terminal / VS Code / Cursor and run this test again.")
    elif max_amp < 0.0001:
        print("⚠️ WARNING: EXTREMELY FAINT OR NO SIGNAL DETECTED")
        print("The device is capturing, but the sound level is virtually zero. The audio is not being routed correctly.")
        print("\n💡 ACTION REQUIRED:")
        print("1. Open 'Audio MIDI Setup' (การตั้งค่า MIDI ออดิโอ) on your Mac.")
        print("2. Select your 'Multi-Output Device' (อุปกรณ์เอาท์พุตหลายเครื่อง) on the left.")
        print("3. Ensure that BOTH 'BlackHole 2ch' and your physical speakers/headphones are checked (ติ๊กถูกทั้งสองอัน).")
        print("4. Make sure your system output (or Teams output setting) is set to 'Multi-Output Device' and not directly to your speakers.")
        print("5. Double-check that 'BlackHole 2ch' (input/output) volumes are not muted or slider is at zero.")
    else:
        print("🎉 SUCCESS! Signal detected successfully.")
        print(f"Sound is coming through. (Peak: {max_amp:.4f}, RMS: {rms:.4f})")
        print("If the translator level meter still does not move:")
        print("1. Check that you selected the correct 'AUDIO INPUT' in the translator web interface dropdown (top-left).")
        print("2. Try reloading the web page.")

if __name__ == "__main__":
    main()
