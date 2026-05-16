import random

class PlayerProfile:
    """
    Player attributes modifying core logic.
    Values from 0 to 100.
    """
    def __init__(self, role_str="CM"):
        # Base stats
        self.pace = random.randint(60, 90)
        self.stamina = random.randint(70, 95)
        self.passing = random.randint(60, 90)
        self.vision = random.randint(50, 85)
        self.shooting = random.randint(50, 85)
        self.composure = random.randint(60, 90)
        
        self.current_stamina = 100.0
        
        # Adjust based on tactical role
        if role_str in ["LW", "RW", "ST"]:
            self.pace += 10
            self.shooting += 10
            self.composure += 5
        elif role_str in ["CDM", "CB", "LB", "RB"]:
            self.pace -= 5
            self.stamina += 10
            self.passing += 5
            
        self.pace = min(100, self.pace)
        self.shooting = min(100, self.shooting)
        
    def get_current_speed_mult(self):
        """Drops to 70% effectiveness at 0 stamina."""
        return 0.7 + 0.3 * (self.current_stamina / 100.0)
