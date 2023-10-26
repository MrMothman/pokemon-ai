from pyboy import PyBoy

def run_game(rom_path):
    pyboy = PyBoy(rom_path)
    
    while not pyboy.tick():
        # Implement your game logic here
        pass

if __name__ == "__main__":
    rom_path = "ROM/pc.gbc"
    run_game(rom_path)