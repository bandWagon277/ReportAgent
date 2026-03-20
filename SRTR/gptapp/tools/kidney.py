
from typing import Any, Dict, List, Optional, Tuple


def calculate_kidney_waiting_time(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate predicted kidney transplant waiting time.
    
    This is a simplified demo implementation based on SRTR's calculator.
    Real implementation would use SRTR's actual statistical models.
    
    Parameters:
    - blood_type: "O", "A", "B", "AB"
    - age: int (years)
    - dialysis_time: int (months on dialysis)
    - cpra: int (0-100, calculated Panel Reactive Antibody percentage)
    - diabetes: bool
    - region: int (1-11, UNOS region)
    
    Returns:
    {
      "median_wait_days": int,
      "wait_ranges": {
        "25th_percentile": int,
        "75th_percentile": int
      },
      "factors_impact": {...},
      "calculation_notes": str
    }
    """
    
    # Validate required parameters
    required = ["blood_type", "age", "dialysis_time", "cpra"]
    missing = [p for p in required if p not in params]
    if missing:
        return {
            "error": f"Missing required parameters: {', '.join(missing)}",
            "required_parameters": {
                "blood_type": "O, A, B, or AB",
                "age": "Patient age in years",
                "dialysis_time": "Months on dialysis",
                "cpra": "CPRA percentage (0-100)"
            }
        }
    
    # Extract and validate parameters
    try:
        blood_type = params["blood_type"].upper()
        age = int(params["age"])
        dialysis_time = int(params["dialysis_time"])
        cpra = int(params["cpra"])
        diabetes = params.get("diabetes", False)
        region = params.get("region", 5)  # Default to region 5
        
        if blood_type not in ["O", "A", "B", "AB"]:
            raise ValueError("Invalid blood type")
        if not (0 <= age <= 120):
            raise ValueError("Age must be between 0-120")
        if not (0 <= cpra <= 100):
            raise ValueError("CPRA must be between 0-100")
        if not (1 <= region <= 11):
            raise ValueError("Region must be between 1-11")
            
    except (ValueError, TypeError) as e:
        return {"error": f"Invalid parameter value: {str(e)}"}
    
    # Base waiting time by blood type (in days)
    # Based on simplified SRTR data - O type waits longest
    base_wait = {
        "O": 1825,   # ~5 years
        "A": 1095,   # ~3 years
        "B": 1460,   # ~4 years
        "AB": 730    # ~2 years
    }
    
    median_wait = base_wait[blood_type]
    
    # Adjustment factors
    factors_impact = {}
    
    # Age factor: Younger patients typically wait longer (more competition)
    if age < 18:
        age_multiplier = 1.3
        factors_impact["age"] = {
            "category": "Pediatric (<18)",
            "impact": "+30% wait time",
            "reason": "Smaller donor pool, prioritized for pediatric donors"
        }
    elif age < 50:
        age_multiplier = 1.0
        factors_impact["age"] = {
            "category": "Adult (18-49)",
            "impact": "Baseline",
            "reason": "Standard allocation priority"
        }
    else:
        age_multiplier = 0.85
        factors_impact["age"] = {
            "category": "Senior (50+)",
            "impact": "-15% wait time",
            "reason": "May accept older/ECD donors"
        }
    
    median_wait *= age_multiplier
    
    # CPRA factor: Higher CPRA = harder to match = longer wait
    if cpra >= 98:
        cpra_multiplier = 2.5
        factors_impact["cpra"] = {
            "value": cpra,
            "category": "Highly sensitized (98-100%)",
            "impact": "+150% wait time",
            "reason": "Very difficult to find compatible donor"
        }
    elif cpra >= 80:
        cpra_multiplier = 1.8
        factors_impact["cpra"] = {
            "value": cpra,
            "category": "Sensitized (80-97%)",
            "impact": "+80% wait time",
            "reason": "Limited compatible donors"
        }
    elif cpra >= 20:
        cpra_multiplier = 1.2
        factors_impact["cpra"] = {
            "value": cpra,
            "category": "Moderately sensitized (20-79%)",
            "impact": "+20% wait time",
            "reason": "Some limitations in donor matching"
        }
    else:
        cpra_multiplier = 1.0
        factors_impact["cpra"] = {
            "value": cpra,
            "category": "Not sensitized (<20%)",
            "impact": "Baseline",
            "reason": "Broad donor compatibility"
        }
    
    median_wait *= cpra_multiplier
    
    # Dialysis time factor: Waiting time credit
    if dialysis_time >= 36:
        dialysis_benefit = 0.9  # 10% reduction for long waiters
        factors_impact["dialysis_time"] = {
            "value": f"{dialysis_time} months",
            "impact": "-10% wait time",
            "reason": "Waiting time priority (36+ months)"
        }
    else:
        dialysis_benefit = 1.0
        factors_impact["dialysis_time"] = {
            "value": f"{dialysis_time} months",
            "impact": "No adjustment",
            "reason": "Less than 36 months"
        }
    
    median_wait *= dialysis_benefit
    
    # Diabetes factor: Slight increase due to medical complexity
    if diabetes:
        median_wait *= 1.1
        factors_impact["diabetes"] = {
            "value": "Yes",
            "impact": "+10% wait time",
            "reason": "May require more careful donor matching"
        }
    
    # Regional variation (simplified)
    regional_multipliers = {
        1: 1.1,   # Region 1 (Northeast) - longer waits
        2: 1.05,
        3: 0.95,
        4: 0.9,
        5: 1.0,   # Baseline
        6: 0.95,
        7: 0.9,
        8: 1.0,
        9: 1.05,
        10: 0.95,
        11: 1.1
    }
    
    regional_mult = regional_multipliers.get(region, 1.0)
    median_wait *= regional_mult
    
    if regional_mult != 1.0:
        factors_impact["region"] = {
            "value": f"Region {region}",
            "impact": f"{'+' if regional_mult > 1 else ''}{int((regional_mult - 1) * 100)}% wait time",
            "reason": "Regional supply/demand variation"
        }
    
    # Calculate ranges (25th and 75th percentiles)
    # Simplified: ±40% from median
    percentile_25 = int(median_wait * 0.6)
    percentile_75 = int(median_wait * 1.4)
    median_wait = int(median_wait)
    
    # Convert to human-readable format
    def days_to_readable(days):
        years = days // 365
        remaining_days = days % 365
        months = remaining_days // 30
        
        if years > 0:
            if months > 0:
                return f"{years} year{'s' if years > 1 else ''}, {months} month{'s' if months > 1 else ''}"
            return f"{years} year{'s' if years > 1 else ''}"
        elif months > 0:
            return f"{months} month{'s' if months > 1 else ''}"
        else:
            return f"{days} days"
    
    return {
        "median_wait_days": median_wait,
        "median_wait_readable": days_to_readable(median_wait),
        "wait_ranges": {
            "25th_percentile_days": percentile_25,
            "25th_percentile_readable": days_to_readable(percentile_25),
            "median_days": median_wait,
            "median_readable": days_to_readable(median_wait),
            "75th_percentile_days": percentile_75,
            "75th_percentile_readable": days_to_readable(percentile_75),
        },
        "patient_profile": {
            "blood_type": blood_type,
            "age": age,
            "dialysis_time_months": dialysis_time,
            "cpra": cpra,
            "diabetes": diabetes,
            "region": region
        },
        "factors_impact": factors_impact,
        "interpretation": generate_wait_time_interpretation(median_wait, factors_impact),
        "calculation_notes": (
            "This is a simplified demonstration model based on SRTR data trends. "
            "Actual waiting times vary significantly based on many factors. "
            "For official estimates, please use SRTR's online calculator at https://www.srtr.org/tools/"
        ),
        "disclaimer": "This calculation is for educational purposes only and should not be used for medical decision-making."
    }

def generate_wait_time_interpretation(median_days: int, factors: Dict) -> str:
    """Generate human-readable interpretation of waiting time calculation."""
    
    years = median_days / 365
    
    if years < 1:
        timeline = "less than a year"
    elif years < 2:
        timeline = "about 1-2 years"
    elif years < 3:
        timeline = "about 2-3 years"
    elif years < 5:
        timeline = "about 3-5 years"
    else:
        timeline = "5 years or longer"
    
    # Identify key factors
    key_factors = []
    for factor_name, factor_data in factors.items():
        impact = factor_data.get("impact", "")
        if "+" in impact and factor_name != "age":  # Factors that increase wait
            key_factors.append(factor_data.get("reason", ""))
    
    interpretation = f"Based on the provided information, the estimated median waiting time is **{timeline}**."
    
    if key_factors:
        interpretation += f" The main factors extending this wait time are: {'; '.join(key_factors[:3])}."
    
    interpretation += " Remember that 25% of similar patients receive transplants sooner, and 25% wait longer than this median estimate."
    
    return interpretation