"""
Personal Digital Twin Assistant - Chat naturally with your AI
Auto-saves conversations and builds long-term memory
"""

import os
from datetime import datetime
from memory_system.local_llm import create_local_system
from memory_system import MemorySystem


def print_header():
    """Print minimal header."""
    print("\n" + "="*60)
    print("Your Digital Twin Assistant")
    print("="*60)
    print("Chat naturally - I'll remember everything important")
    print("Type 'exit' or 'quit' to end conversation")
    print("="*60 + "\n")


def main():
    """Personal assistant chat loop."""
    
    # Initialize local system
    print("Connecting to your local Llama model...")
    
    try:
        llm_callable, embedding_service, dimension = create_local_system(
            model="llama3.1:8b-instruct-q4_K_M",
            base_url="http://localhost:11434"
        )
        
        print()
        system = MemorySystem(
            llm_callable=llm_callable,
            embedding_service=embedding_service,
            storage_path="./assistant_memory"
        )
        
    except Exception as e:
        print(f"\nFailed to connect: {e}")
        print("\nMake sure Ollama is running:")
        print("   In WSL: ollama serve")
        return
    
    print_header()
    
    # Get user's name if first time
    stats = system.get_stats()
    if stats['vector_store']['total_memories'] == 0:
        print("Hi! I'm your new digital twin assistant.")
        print("What should I call you?\n")
        name = input("You: ").strip()
        if name:
            print(f"\nNice to meet you, {name}! Let's chat.\n")
            # Process this as first memory
            system.chat(f"My name is {name}", llm_callable)
            system.process_memories(chunk_size=1, save_to_file=False)
    
    turn_count = 0
    auto_save_threshold = 5  # Save and process every 5 messages
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Exit commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\nAssistant: Talk to you later!")
                
                # Save final conversation
                if turn_count > 0:
                    print("\nSaving our conversation...")
                    try:
                        system.process_memories(save_to_file=True)
                        print("[OK] All memories saved!\n")
                    except Exception as e:
                        print(f"[WARN] Couldn't process memories: {e}")
                        print("Conversation saved, but memory extraction failed.\n")
                break
            
            # Manual commands (hidden)
            if user_input == "/save":
                filepath = system.save_conversation_to_file()
                print(f"\nSaved to: {os.path.basename(filepath)}\n")
                continue
            
            if user_input == "/memories":
                print()
                system.list_memories(limit=10)
                continue
            
            if user_input == "/stats":
                stats = system.get_stats()
                print(f"\n[STATS] Memories: {stats['vector_store']['total_memories']}")
                print(f"[STATS] Current chat: {stats['current_session_turns']} turns\n")
                continue
            
            # Normal conversation
            print("Assistant: ", end="", flush=True)
            response = system.chat(user_input, llm_callable)
            print(response)
            
            turn_count += 1
            
            # Auto-save and process every N turns
            if turn_count >= auto_save_threshold:
                print(f"\n[Processing {turn_count} messages into long-term memory...]")
                memories_added = system.process_memories(save_to_file=True)
                if memories_added > 0:
                    print(f"[OK] Learned {memories_added} new things about you")
                turn_count = 0  # Reset counter
        
        except KeyboardInterrupt:
            print("\n\nAssistant: Catch you later!\n")
            if turn_count > 0:
                try:
                    system.process_memories(save_to_file=True)
                except Exception as e:
                    print(f"[WARN] Couldn't process memories (Ollama disconnected): {e}")
                    print("Your conversation was saved, memories will process next time.")
            break
        
        except Exception as e:
            print(f"\n[ERROR] {e}")
            print("Let's keep chatting though!\n")


if __name__ == "__main__":
    main()
us 