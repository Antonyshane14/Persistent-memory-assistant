"""
Hunter Alpha-Powered Assistant - Uses Hunter Alpha via OpenRouter for chat and memory
100% Hunter Alpha - reliable, intelligent, perfect JSON parsing via OpenRouter.
"""

import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
from datetime import datetime
from dotenv import load_dotenv
from memory_system.claude_llm import create_claude_system
from memory_system import MemorySystem


def print_header():
    """Print minimal header."""
    print("\n" + "="*60)
    print("Your Digital Twin Assistant (Hunter Alpha Edition)")
    print("="*60)
    print("💬 Powered by Hunter Alpha via OpenRouter")
    print("🧠 Smart memory extraction")
    print("="*60)
    print("Chat naturally - I'll remember everything important")
    print("Type 'exit' or 'quit' to end conversation")
    print("="*60 + "\n")


def main():
    """Claude-powered assistant for both chat and memory."""

    # Load environment variables
    load_dotenv()

    # Initialize Hunter Alpha via OpenRouter
    print("🔧 Connecting to Hunter Alpha via OpenRouter...")
    try:
        claude_llm, claude_embedding, dimension = create_claude_system(
            model="hunter-alpha"
        )
    except Exception as e:
        print(f"\n❌ Failed to connect to OpenRouter: {e}")
        print("\nMake sure OPENROUTER_API_KEY is set in .env file")
        return

    # Initialize memory system with Claude
    print("\n🔧 Initializing memory system...")
    try:
        system = MemorySystem(
            llm_callable=claude_llm,  # Claude for memory extraction
            embedding_service=claude_embedding,  # Mock embeddings
            storage_path="./assistant_memory"
        )
    except Exception as e:
        print(f"\n❌ Failed to initialize memory system: {e}")
        import traceback
        traceback.print_exc()
        return

    print_header()

    # Get user's name if first time
    stats = system.get_stats()
    if stats['vector_store']['total_memories'] == 0:
        print("Hi! I'm your new digital twin assistant powered by Hunter Alpha.")
        print("What should I call you?\n")
        name = input("You: ").strip()
        if name:
            print(f"\nNice to meet you, {name}! Let's chat.\n")
            # Process this as first memory
            response = system.chat(f"My name is {name}", claude_llm)
            print(f"Assistant: {response}")
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
                    print("\n💾 Saving our conversation...")
                    try:
                        system.process_memories(save_to_file=True)
                        print("✓ All memories saved!\n")
                    except Exception as e:
                        print(f"⚠️  Couldn't process memories: {e}")
                        print("Conversation saved, but memory extraction failed.\n")
                break

            # Manual commands
            if user_input == "/save":
                filepath = system.save_conversation_to_file()
                print(f"\n💾 Saved to: {os.path.basename(filepath)}\n")
                continue

            if user_input == "/memories":
                print()
                system.list_memories(limit=10)
                continue

            if user_input == "/stats":
                stats = system.get_stats()
                print(f"\n📊 Stats:")
                print(f"   Memories: {stats['vector_store']['total_memories']}")
                print(f"   Current chat: {stats['current_session_turns']} turns\n")
                continue

            # Normal conversation - use Claude for chat
            print("Assistant: ", end="", flush=True)
            response = system.chat(user_input, claude_llm)
            print(response)

            turn_count += 1

            # Auto-save and process every N turns
            if turn_count >= auto_save_threshold:
                print(f"\n💾 [Processing {turn_count} messages into long-term memory...]")
                try:
                    memories_added = system.process_memories(save_to_file=True)
                    if memories_added > 0:
                        print(f"✓ Learned {memories_added} new things about you\n")
                    else:
                        print(f"✓ Conversation saved\n")
                    turn_count = 0  # Reset counter
                except Exception as e:
                    print(f"⚠️  Memory processing error: {e}")
                    print("Chat continues, memories will be retried later.\n")

        except KeyboardInterrupt:
            print("\n\nAssistant: Catch you later!\n")
            if turn_count > 0:
                try:
                    system.process_memories(save_to_file=True)
                except Exception as e:
                    print(f"⚠️  Couldn't process memories: {e}")
                    print("Your conversation was saved.")
            break

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print("\nLet's keep chatting though!\n")


if __name__ == "__main__":
    main()
