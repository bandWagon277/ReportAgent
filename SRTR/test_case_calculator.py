"""
Test script for Kidney Waiting Time Calculator

Usage:
    python test_calculator.py
"""

import requests
import json

API_BASE = "http://localhost:8000"

def test_query(prompt, description=""):
    """Test the unified query endpoint"""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"{'='*60}")
    print(f"Query: {prompt}")
    print()
    
    response = requests.post(
        f"{API_BASE}/api/query",
        json={"prompt": prompt},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        data = response.json()
        
        print("✅ SUCCESS")
        print(f"\nIntent: {data.get('plan', {}).get('intent')}")
        print(f"Answer Mode: {data.get('plan', {}).get('answer_mode')}")
        print(f"\nSummary: {data.get('summary')}")
        try:
            print(f"\nDetail: {data.get('detail')}")
        except:
            print("No details for this mode.")
        
        if data.get('key_points'):
            print(f"\nKey Points:")
            for point in data['key_points']:
                print(f"  • {point}")
        
        if data.get('confidence') == 'incomplete':
            print(f"\n⚠️  Missing: {data.get('missing_parameters')}")
        
        if data.get('calculation_result'):
            result = data['calculation_result']
            print(f"\n📊 Calculation Result:")
            print(f"  Median Wait: {result['median_wait_readable']}")
            print(f"  Range: {result['wait_ranges']['25th_percentile_readable']} to {result['wait_ranges']['75th_percentile_readable']}")
    else:
        print(f"❌ FAILED: {response.status_code}")
        print(response.text)
    
    return response

def test_direct_calculator(params, description=""):
    """Test the direct calculator endpoint"""
    print(f"\n{'='*60}")
    print(f"DIRECT CALCULATOR TEST: {description}")
    print(f"{'='*60}")
    print(f"Parameters: {json.dumps(params, indent=2)}")
    print()
    
    response = requests.post(
        f"{API_BASE}/api/calculate",
        json={
            "tool": "kidney_waiting_time",
            "parameters": params
        },
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        data = response.json()
        result = data['result']
        
        print("✅ SUCCESS")
        print(f"\nMedian Wait: {result['median_wait_readable']}")
        print(f"Range: {result['wait_ranges']['25th_percentile_readable']} to {result['wait_ranges']['75th_percentile_readable']}")
        
        print(f"\n🎯 Factors Impact:")
        for factor, details in result['factors_impact'].items():
            print(f"  • {factor.replace('_', ' ').title()}: {details['impact']}")
    else:
        print(f"❌ FAILED: {response.status_code}")
        print(response.text)
    
    return response

def run_all_tests():
    """Run comprehensive test suite"""
    
    print("\n" + "="*60)
    print("KIDNEY WAITING TIME CALCULATOR - TEST SUITE")
    print("="*60)
    
    # Test 1: Complete natural language query
    test_query(
        "I'm 45 years old, type O blood, been on dialysis for 2 years, and my CPRA is 85. How long will I wait?",
        "Complete parameters in natural language"
    )
    
    # Test 2: Look up for definitions
    test_query(
        "What is DONOR_ID?",
        "dictionary lookup"
    )
    
    # Test 3: Highly sensitized patient
    test_query(
        "Calculate wait time: 50 year old, type O, 4 years on dialysis, CPRA 99, diabetic",
        "Highly sensitized patient"
    )
    
    # Test 4: Low-wait scenario
    test_query(
        "I'm 35, blood type AB, 6 months dialysis, CPRA is 5",
        "Favorable scenario (AB, low CPRA)"
    )
    
    # Test 5: Pediatric patient
    test_query(
        "My child is 12 years old, type A, no dialysis yet, CPRA 0",
        "Pediatric patient"
    )
    
    # Test 6: Question about methodology (should NOT trigger calculator)
    test_query(
        "How is kidney waiting time calculated?",
        "Methodology question (not calculator)"
    )

        
    """# Test 2: Missing parameters
    test_query(
        "I'm 30 years old, type AB. Calculate my kidney waiting time.",
        "Missing dialysis_time and CPRA"
    )"""
    
    """# Test 7: Direct calculator call
    test_direct_calculator(
        {
            "blood_type": "O",
            "age": 45,
            "dialysis_time": 24,
            "cpra": 85,
            "diabetes": False,
            "region": 5
        },
        "Direct API call with structured parameters"
    )"""
    
    """# Test 8: Invalid blood type
    test_direct_calculator(
        {
            "blood_type": "XYZ",
            "age": 30,
            "dialysis_time": 12,
            "cpra": 20
        },
        "Invalid blood type (should error)"
    )
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)"""

if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to server at", API_BASE)
        print("Please make sure Django is running: python manage.py runserver")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()