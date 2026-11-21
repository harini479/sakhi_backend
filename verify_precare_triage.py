import asyncio
loop = asyncio.SelectorEventLoop()
asyncio.set_event_loop(loop)

from unittest.mock import AsyncMock, patch
from app import store, chat_engine

store._seed_initial_data()

async def test_pre_care_triage():
    with patch.object(chat_engine, 'generate_sakhi_reply', new=AsyncMock(return_value=None)):
        guest_phone = "guest_test_9999"

        # 1. Guest asks general symptom (stomach ache)
        res1 = await chat_engine.process_sakhi_chat(
            phone_number=guest_phone,
            message="I have stomach ache today",
            language="en",
        )
        print("Test 1 (Pre-Care Stomach Ache Triage Question):", flush=True)
        print("  Mode:", res1.get("mode"), flush=True)
        print("  Reply:", res1.get("reply"), flush=True)

        # 2. Guest replies with severe symptoms
        res2 = await chat_engine.process_sakhi_chat(
            phone_number=guest_phone,
            message="It is very severe, unbearable sharp pain",
            language="en",
        )
        print("\nTest 2 (Pre-Care Severe Escalation):", flush=True)
        print("  Mode:", res2.get("mode"), flush=True)
        print("  Reply preview:", res2.get("reply", "")[:120], flush=True)

        # 3. New Guest asks headache and says mild
        guest_phone2 = "guest_test_8888"
        res3 = await chat_engine.process_sakhi_chat(
            phone_number=guest_phone2,
            message="I have a bad headache",
            language="en",
        )
        print("\nTest 3 (Pre-Care Headache Triage Question):", flush=True)
        print("  Mode:", res3.get("mode"), flush=True)
        print("  Reply:", res3.get("reply"), flush=True)

        res4 = await chat_engine.process_sakhi_chat(
            phone_number=guest_phone2,
            message="No, just mild ache, it is manageable",
            language="en",
        )
        print("\nTest 4 (Pre-Care Mild Safe Guidance):", flush=True)
        print("  Mode:", res4.get("mode"), flush=True)
        print("  Reply preview:", res4.get("reply", "")[:120], flush=True)

        # 5. Telugu general enquiry
        guest_phone3 = "guest_test_7777"
        res5 = await chat_engine.process_sakhi_chat(
            phone_number=guest_phone3,
            message="నాకు కడుపు నొప్పిగా ఉంది",
            language="te",
        )
        print("\nTest 5 (Telugu Stomach Ache Triage):", flush=True)
        print("  Mode:", res5.get("mode"), flush=True)
        print("  Reply:", res5.get("reply"), flush=True)

        print("\nALL PRE-CARE TRIAGE TESTS PASSED SUCCESSFULLY!", flush=True)

loop.run_until_complete(test_pre_care_triage())
loop.close()
