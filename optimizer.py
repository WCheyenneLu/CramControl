import numpy as np
from scipy.optimize import linprog
from datetime import date, datetime
import copy

def get_week_index(due_iso: str, quarter_start: date) -> int:
    """Returns an integer 1-11 representing the week index (11 = Finals)."""
    if not due_iso:
        return -1
    try:
        due_date = datetime.fromisoformat(str(due_iso)).date()
        days_diff = (due_date - quarter_start).days
        
        # FIX: If it's more than 14 days before the quarter, it's likely a wrong year parsed.
        if days_diff < -14: 
            return -1 
        elif days_diff < 0: 
            return 1 # Normal early prep (e.g., read syllabus before day 1)
            
        week_num = (days_diff // 7) + 1
        return min(week_num, 11)
    except Exception:
        return -1

def get_lead_time_weeks(task_type: str) -> int:
    """
    Capped at a maximum of 2 weeks lead time for all tasks, 
    as requested to prevent overly stretched schedules.
    """
    lead_times = {
        "reading": 0,          # Same week
        "weekly_short": 0,     # Same week
        "weekly_long": 0,      # Same week
        "essay": 1,            # 1 week before
        "midterm_easy": 1,     # 1 week before
        "project": 1,          # 1 week before
        "final_hard": 2,       # Max 2 weeks before
        "capstone": 2,         # Max 2 weeks before
        "other": 0
    }
    return lead_times.get(task_type, 0)

def optimize_weekly_schedule(items: list[dict], quarter_start: date) -> dict:
    """
    Splits tasks across their realistic working windows and optimizes each week.
    """
    # 1. Initialize Weekly Buckets
    weekly_schedule = {
        f"Week {i}": {"tasks": [], "optimal_allocation": {}} for i in range(1, 11)
    }
    weekly_schedule["Finals Week"] = {"tasks": [], "optimal_allocation": {}}
    weekly_schedule["Unscheduled"] = {"tasks": [], "optimal_allocation": {}}

    # 2. Bucket and Split the items
    for item in items:
        due_idx = get_week_index(item.get("due_date_iso"), quarter_start)
        
        if due_idx == -1:
            weekly_schedule["Unscheduled"]["tasks"].append(item)
            continue
            
        # Determine the working window
        task_type = item.get("task_type", "other")
        lead_weeks = get_lead_time_weeks(task_type)
        
        # Floor the start week at Week 1 so we don't schedule before the quarter begins
        start_idx = max(1, due_idx - lead_weeks)
        span = (due_idx - start_idx) + 1 # Number of weeks this task spans
        
        # Split the estimated hours evenly across the weeks
        original_est = item.get("estimated_hours", 2.0)
        split_est = original_est / span
        
        # Distribute chunks of the task into the respective weeks
        for w in range(start_idx, due_idx + 1):
            week_key = f"Week {w}" if w <= 10 else "Finals Week"
            
            # Deep copy so we don't accidentally overwrite the original object references
            task_chunk = copy.deepcopy(item)
            task_chunk["estimated_hours"] = split_est
            
            # Rename if it spans multiple weeks so the user knows it's a chunk
            if span > 1:
                part_num = w - start_idx + 1
                task_chunk["assignment_name"] = f"{item['assignment_name']} (Part {part_num}/{span})"
                
            weekly_schedule[week_key]["tasks"].append(task_chunk)

    # 3. Optimize each week using scipy.optimize.linprog
    for week_name, data in weekly_schedule.items():
        tasks = data["tasks"]
        
        if not tasks and week_name == "Unscheduled":
            continue

        c = []
        bounds = []
        task_names = []

        # A. Setup Tasks bounds and priorities
        for i, task in enumerate(tasks):
            est = task.get("estimated_hours", 2.0)
            weight = task.get("weight_numeric", 5.0)
            if est <= 0: est = 1.0
            
            priority = -(weight) if weight > 0 else -2.0
            c.append(priority)
            
            # THE FIX: Lower the floor from 0.5 down to 0.1
            # This allows the solver to squeeze task times during "hell weeks" 
            # instead of crashing and returning 0 for everything.
            bounds.append((est * 0.1, est * 1.5)) 
            task_names.append(task.get("assignment_name", f"Task {i}"))

        # B. Setup General Life bounds (7 day basis)
        c.extend([-5, -3]) 
        bounds.append((7, 21)) # Self Care: 1-3 hrs/day
        bounds.append((0, 35)) # Social: 0-5 hrs/day

        # C. Constraints (Inequality: Aub * x <= bub)
        total_weekly_hours = 91 # 13 free hours/day * 7 days
        A_ub = [[1] * len(c)]
        b_ub = [total_weekly_hours]

        # D. Run Optimizer
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        # E. Save Results
        if res.success:
            allocation = {"general_life": {}, "specific_tasks": {}}
            for i, task_name in enumerate(task_names):
                opt_hours = round(res.x[i], 1)
                allocation["specific_tasks"][task_name] = opt_hours
                tasks[i]["optimal_hours"] = opt_hours 

            allocation["general_life"]["self_care"] = round(res.x[-2], 1)
            allocation["general_life"]["social"] = round(res.x[-1], 1)
            
            data["optimal_allocation"] = allocation
            data["optimization_status"] = "Success"
        else:
            data["optimization_status"] = "Infeasible (Too much work, bounds failed)"

    return weekly_schedule