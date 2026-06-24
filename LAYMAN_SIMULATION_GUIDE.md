# 🚀 Layman's Guide: Simulating TinyML on RISC-V

If you're new to specialized hardware or "TinyML," all those commands can look like magic. Here is a simple breakdown of what we just did and why each part matters.

---

## 🏗️ The Big Picture
Imagine you have a brain (your **AI Model**) that you want to put into a tiny robot (a **RISC-V Chip**). 

However, you don't actually have the physical robot yet. So, you use a high-tech video game (the **Spike Simulator**) to pretend you have the robot on your computer.

### The "Players" in this Story:
1.  **TinyML (The Brain)**: A very small version of an AI model that can fit on tiny computers.
2.  **RISC-V (The Language)**: A specific set of instructions that the tiny computer understands. Think of it like a dialect of human language.
3.  **Spike (The Virtual World)**: A software that mimics a RISC-V processor on your normal computer.
4.  **The Proxy Kernel (The Translator)**: Since the "Virtual World" is a bare-bones environment, the Proxy Kernel (PK) acts as a middleman that lets the simulator talk to your terminal (to print text).

---

## 🛠️ What We Actually Did (In Plain English)

### 1. Preparing the Brain (`model_data.cc`)
Computers like to read data in binary (1s and 0s). We took your saved AI model and turned it into a massive list of numbers that we can "bake" directly into the software.

### 2. Building the Engine (TensorFlow Lite Micro)
Even a simple brain needs an engine to function. We downloaded **TensorFlow Lite Micro**, which is the "nervous system" that knows how to run your AI brain on small chips. We had to "compile" it, which is like building a car from a blueprint so it's ready to drive.

### 3. Writing the Instructions (`main.cc`)
Think of this as the "Mission Control." It tells the simulator:
- "Here is the brain (the model)."
- "Here is some memory to think in."
- "Take this data sample, run it through the brain, and tell me what activity it is (Walking, Sitting, etc.)."

### 4. Translation & Simulation
We used a **Cross-Compiler**. This is a special tool that takes your C++ code and translates it into the **RISC-V language**. 
Normally, your computer speaks "x86," but our simulator only speaks "RISC-V." If we didn't use a cross-compiler, the simulator wouldn't understand a single word.

---

## 🚦 How to Run It (The "Cheat Sheet")

If you want to run it again, just remember these 3 steps:

1.  **Set the Stage**: Tell the computer where your tools are.
    ```bash
    export PATH=/home/the-stick-insect/riscv-tools/bin:$PATH
    ```
2.  **Build It**: Translate the code into a "RISC-V" file.
    ```bash
    # (The long 'riscv-none-elf-g++' command from the main guide)
    ```
3.  **Simulate It**: Open the virtual world and run the file.
    ```bash
    spike pk tinyml_sim_exec
    ```

---

## ❓ Why did it "Segfault" (Crash) earlier?
Computers are very picky about where they store data. Imagine a bookshelf where every book MUST start at a specific numbered slot. 
One of our "books" was slightly off-place, so when the simulator tried to read it, it got confused and crashed. We fixed this by telling the computer to "Align" the data perfectly on its shelf.

---

**Next Steps**: If you want to change the class it predicts, you can extract a different sample from your dataset and put it into the code!
