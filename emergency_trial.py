import asyncio
import math
import os
import platform
import random
import sys
import threading
import time
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import pygame

# Set modern plotting style (from test3.py)
plt.style.use('seaborn-v0_8')
import seaborn as sns
sns.set_palette("husl")

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
from collections import deque

# ============================================================================
# CONFIGURATION AND CONSTANTS (Enhanced from test3.py)
# ============================================================================

class Config:
    """Global configuration settings for the simulation."""
    
    # Signal timing defaults (seconds)
    DEFAULT_RED = 150
    DEFAULT_YELLOW = 5
    DEFAULT_GREEN = 25  # Reduced default for more responsiveness (from test3.py)
    DEFAULT_MINIMUM = 8  # Reduced minimum (from test3.py)
    DEFAULT_MAXIMUM = 50  # Reduced maximum (from test3.py)
    
    # Simulation parameters
    NUM_SIGNALS = 4
    NUM_LANES = 3
    SIMULATION_TIME = 200  # Keep original, but can extend if needed
    
    # Vehicle parameters (Enhanced from test3.py)
    VEHICLE_TIMES = {
        'car': 1.8,      # Faster processing
        'bike': 0.8,
        'rickshaw': 2.0,
        'bus': 2.2,
        'truck': 2.4
    }
    
    VEHICLE_SPEEDS = {
        'car': 2.5,      # Increased speeds
        'bus': 2.0,
        'truck': 2.0,
        'rickshaw': 2.2,
        'bike': 2.8
    }
    
    # Enhanced fairness limits (increased thresholds to prioritize adaptive) (from test3.py)
    FAIRNESS_LIMITS = {0: 8, 1: 7, 2: 8, 3: 7}  # Increased from {5,3,4,3}
    
    # Adaptive priority weights (from test3.py)
    ADAPTIVE_WEIGHT = 0.85     # 85% preference for adaptive
    FAIRNESS_WEIGHT = 0.15     # 15% for fairness
    
    # Display settings
    SCREEN_WIDTH = 1400
    SCREEN_HEIGHT = 820
    FPS = 60
    
    # Vehicle spacing (increased for better collision avoidance)
    STOPPING_GAP = 18  # From test3.py
    MOVING_GAP = 22    # From test3.py
    ROTATION_ANGLE = 3


# ============================================================================
# COORDINATE SYSTEM AND LAYOUT
# ============================================================================

class IntersectionLayout:
    """Defines the intersection layout and coordinates."""
    
    # Starting positions for vehicles by direction and lane
    START_X = {
        'right': [0, 0, 0],
        'down': [755, 727, 697],
        'left': [1400, 1400, 1400],
        'up': [602, 627, 657]
    }
    
    START_Y = {
        'right': [348, 370, 398],
        'down': [0, 0, 0],
        'left': [498, 466, 436],
        'up': [800, 830, 860]
    }
    
    # Signal display coordinates
    SIGNAL_COORDS = [(530, 230), (810, 230), (810, 570), (530, 570)]
    SIGNAL_TIMER_COORDS = [(530, 210), (810, 210), (810, 550), (530, 550)]
    VEHICLE_COUNT_COORDS = [(480, 210), (880, 210), (880, 550), (480, 550)]
    
    # Stop line positions
    STOP_LINES = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
    DEFAULT_STOPS = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
    
    # Turning coordinates
    MID_POINTS = {
        'right': {'x': 705, 'y': 445},
        'down': {'x': 695, 'y': 450},
        'left': {'x': 695, 'y': 425},
        'up': {'x': 695, 'y': 400}
    }


# ============================================================================
# CORE CLASSES (Enhanced with fields from test3.py)
# ============================================================================

class TrafficSignal:
    """Represents a traffic signal with timing control."""
    
    def __init__(self, red: int, yellow: int, green: int, minimum: int, maximum: int):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signal_text = str(green)
        self.total_green_time = 0
        self.activation_count = 0  # From test3.py: Track activations
        self.total_vehicles_served = 0  # From test3.py
    
    def activate(self, green_time: int):  # From test3.py
        """Activate this signal with specified green time."""
        self.green = green_time
        self.activation_count += 1
        self.total_green_time += green_time
    
    def __repr__(self):
        return f"TrafficSignal(R:{self.red}, Y:{self.yellow}, G:{self.green})"


class Vehicle(pygame.sprite.Sprite):
    """Represents a vehicle in the simulation."""
    
    VEHICLE_TYPES = {0: 'car', 1: 'bus', 2: 'truck', 3: 'rickshaw', 4: 'bike'}
    DIRECTION_NUMBERS = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
    
    def __init__(self, lane: int, vehicle_class: str, direction_number: int, 
                 direction: str, will_turn: bool):
        pygame.sprite.Sprite.__init__(self)
        
        # Basic properties
        self.lane = lane
        self.vehicle_class = vehicle_class
        self.speed = Config.VEHICLE_SPEEDS[vehicle_class]
        self.direction_number = direction_number
        self.direction = direction
        self.will_turn = will_turn
        
        # Position and state
        self.x = IntersectionLayout.START_X[direction][lane]
        self.y = IntersectionLayout.START_Y[direction][lane]
        self.crossed = 0
        self.turned = 0
        self.rotate_angle = 0
        
        # Timing (Enhanced from test3.py)
        self.spawn_time = SimulationManager.time_elapsed
        self.cross_time = None
        self.wait_time = 0  # From test3.py
        self.stopped_time = 0  # From test3.py
        self.moving = True  # From test3.py
        
        # Add to vehicle registry
        self._register_vehicle()
        
        # Load or create vehicle image
        self._load_image()
        
        # Set stopping position
        self._set_stop_position()
        
        # Add to simulation
        SimulationManager.simulation.add(self)
    
    def _register_vehicle(self):
        """Register this vehicle in the global vehicle registry."""
        SimulationManager.vehicles[self.direction][self.lane].append(self)
        self.index = len(SimulationManager.vehicles[self.direction][self.lane]) - 1
    
    def _load_image(self):
        """Load vehicle image or create placeholder. (Enhanced fallback from test3.py)"""
        image_path = f"images/{self.direction}/{self.vehicle_class}.png"
        
        if os.path.exists(image_path):
            self.original_image = pygame.image.load(image_path)
            self.current_image = pygame.image.load(image_path)
        else:
            # Enhanced colored placeholders with better sizing
            size_map = {
                'car': (35, 18), 'bus': (50, 22), 'truck': (45, 20),
                'rickshaw': (30, 15), 'bike': (25, 12)
            }
            width, height = size_map.get(self.vehicle_class, (35, 18))
            
            self.original_image = pygame.Surface((width, height))
            # Vehicle type specific colors
            color_map = {
                'car': (100, 150, 255), 'bus': (255, 200, 50), 
                'truck': (150, 100, 50), 'rickshaw': (100, 255, 150),
                'bike': (255, 100, 150)
            }
            color = color_map.get(self.vehicle_class, (150, 150, 150))
            self.original_image.fill(color)
            self.current_image = self.original_image.copy()
    
    def _set_stop_position(self):
        """Calculate and set the stopping position for this vehicle."""
        vehicles_in_lane = SimulationManager.vehicles[self.direction][self.lane]
        gap = Config.STOPPING_GAP
        
        if len(vehicles_in_lane) > 1 and vehicles_in_lane[self.index - 1].crossed == 0:
            prev_vehicle = vehicles_in_lane[self.index - 1]
            if self.direction in ['right', 'down']:
                dimension = (prev_vehicle.current_image.get_rect().width 
                           if self.direction == 'right' 
                           else prev_vehicle.current_image.get_rect().height)
                self.stop = prev_vehicle.stop - dimension - gap
            else:  # left, up
                dimension = (prev_vehicle.current_image.get_rect().width 
                           if self.direction == 'left' 
                           else prev_vehicle.current_image.get_rect().height)
                self.stop = prev_vehicle.stop + dimension + gap
        else:
            self.stop = IntersectionLayout.DEFAULT_STOPS[self.direction]
        
        # Update starting positions for next vehicle
        self._update_spawn_positions()
    
    def _update_spawn_positions(self):
        """Update spawn positions after vehicle placement."""
        dimension = (self.current_image.get_rect().width + Config.STOPPING_GAP 
                    if self.direction in ['right', 'left'] 
                    else self.current_image.get_rect().height + Config.STOPPING_GAP)
        
        if self.direction == 'right':
            IntersectionLayout.START_X[self.direction][self.lane] -= dimension
            SimulationManager.stops[self.direction][self.lane] -= dimension
        elif self.direction == 'left':
            IntersectionLayout.START_X[self.direction][self.lane] += dimension
            SimulationManager.stops[self.direction][self.lane] += dimension
        elif self.direction == 'down':
            IntersectionLayout.START_Y[self.direction][self.lane] -= dimension
            SimulationManager.stops[self.direction][self.lane] -= dimension
        elif self.direction == 'up':
            IntersectionLayout.START_Y[self.direction][self.lane] += dimension
            SimulationManager.stops[self.direction][self.lane] += dimension
    
    def move(self):
        """Update vehicle position based on traffic rules and signal state. (Enhanced tracking from test3.py)"""
        was_moving = self.moving
        
        if self.direction == 'right':
            self._move_right()
        elif self.direction == 'down':
            self._move_down()
        elif self.direction == 'left':
            self._move_left()
        elif self.direction == 'up':
            self._move_up()
        
        # Track stopped time for analytics (from test3.py)
        if not self.moving and not was_moving:
            self.stopped_time += 1
        
        # Update wait time if not crossed yet (from test3.py)
        if self.crossed == 0:
            self.wait_time = SimulationManager.time_elapsed - self.spawn_time
    
    def _move_right(self):
        """Handle movement for vehicles going right."""
        old_x = self.x
        # Check if vehicle has crossed the stop line
        if (self.crossed == 0 and 
            self.x + self.current_image.get_rect().width > IntersectionLayout.STOP_LINES[self.direction]):
            self._mark_crossed()
        
        if self.will_turn:
            self._handle_right_turn()
        else:
            self._move_straight_right()
        
        self.moving = (self.x != old_x)  # From test3.py
    
    def _move_down(self):
        """Handle movement for vehicles going down."""
        old_y = self.y
        if (self.crossed == 0 and 
            self.y + self.current_image.get_rect().height > IntersectionLayout.STOP_LINES[self.direction]):
            self._mark_crossed()
        
        if self.will_turn:
            self._handle_down_turn()
        else:
            self._move_straight_down()
        
        self.moving = (self.y != old_y)  # From test3.py
    
    def _move_left(self):
        """Handle movement for vehicles going left."""
        old_x = self.x
        if self.crossed == 0 and self.x < IntersectionLayout.STOP_LINES[self.direction]:
            self._mark_crossed()
        
        if self.will_turn:
            self._handle_left_turn()
        else:
            self._move_straight_left()
        
        self.moving = (self.x != old_x)  # From test3.py
    
    def _move_up(self):
        """Handle movement for vehicles going up."""
        old_y = self.y
        if self.crossed == 0 and self.y < IntersectionLayout.STOP_LINES[self.direction]:
            self._mark_crossed()
        
        if self.will_turn:
            self._handle_up_turn()
        else:
            self._move_straight_up()
        
        self.moving = (self.y != old_y)  # From test3.py
    
    def _mark_crossed(self):
        """Mark vehicle as having crossed the intersection. (Enhanced from test3.py)"""
        self.crossed = 1
        self.cross_time = SimulationManager.time_elapsed
        SimulationManager.vehicles[self.direction]['crossed'] += 1
        SimulationManager.waiting_times[self.direction_number].append(
            self.cross_time - self.spawn_time
        )
        # Update signal statistics (from test3.py)
        current_signal = SimulationManager.signals[self.direction_number]
        current_signal.total_vehicles_served += 1
    
    def _can_move_forward(self) -> bool:
        """Check if vehicle can move forward based on signal and traffic."""
        # Signal permission check
        signal_allows = (SimulationManager.current_green == self.direction_number and 
                        SimulationManager.current_yellow == 0)
        
        # If already crossed, can always move
        if self.crossed == 1:
            return self._check_intersection_clearance()
        
        # Check stop line adherence
        at_stop_line = False
        if self.direction == 'right':
            at_stop_line = self.x + self.current_image.get_rect().width > self.stop
        elif self.direction == 'down':
            at_stop_line = self.y + self.current_image.get_rect().height > self.stop
        elif self.direction == 'left':
            at_stop_line = self.x < self.stop
        elif self.direction == 'up':
            at_stop_line = self.y < self.stop
        
        # If at stop line and signal doesn't allow, must stop
        if at_stop_line and not signal_allows:
            return False
        
        # Check for vehicle collision ahead in same lane
        return self._check_vehicle_ahead() and self._check_intersection_clearance()
    
    def _check_vehicle_ahead(self) -> bool:
        """Check if there's enough space to move without hitting vehicle ahead. (Enhanced from test3.py)"""
        vehicles_in_lane = SimulationManager.vehicles[self.direction][self.lane]
        
        if self.index == 0:  # First vehicle in lane
            return True
            
        ahead_vehicle = vehicles_in_lane[self.index - 1]
        if ahead_vehicle.crossed == 1 and ahead_vehicle.turned == 1:
            return True  # Vehicle ahead has cleared the lane
        
        safe_distance = Config.MOVING_GAP + 3  # Extra safety margin (from test3.py)
        
        if self.direction == 'right':
            return (self.x + self.current_image.get_rect().width + safe_distance < ahead_vehicle.x)
        elif self.direction == 'down':
            return (self.y + self.current_image.get_rect().height + safe_distance < ahead_vehicle.y)
        elif self.direction == 'left':
            return (self.x > ahead_vehicle.x + ahead_vehicle.current_image.get_rect().width + safe_distance)
        elif self.direction == 'up':
            return (self.y > ahead_vehicle.y + ahead_vehicle.current_image.get_rect().height + safe_distance)
        
        return True
    
    def _check_intersection_clearance(self) -> bool:
        """Check if intersection is clear to avoid collisions with cross-traffic."""
        if not self.crossed:
            return True  # Not in intersection yet
        
        # Define intersection bounds
        intersection_bounds = {
            'left': 590, 'right': 810, 'top': 330, 'bottom': 535
        }
        
        my_rect = pygame.Rect(self.x, self.y, 
                             self.current_image.get_rect().width,
                             self.current_image.get_rect().height)
        
        # Check collision with vehicles from other directions
        for other_direction in ['right', 'down', 'left', 'up']:
            if other_direction == self.direction:
                continue
                
            for lane in range(3):
                for other_vehicle in SimulationManager.vehicles[other_direction][lane]:
                    if other_vehicle == self or other_vehicle.crossed == 0:
                        continue
                    
                    other_rect = pygame.Rect(other_vehicle.x, other_vehicle.y,
                                           other_vehicle.current_image.get_rect().width,
                                           other_vehicle.current_image.get_rect().height)
                    
                    # Check if both vehicles are in intersection area
                    if (my_rect.centerx >= intersection_bounds['left'] and 
                        my_rect.centerx <= intersection_bounds['right'] and
                        my_rect.centery >= intersection_bounds['top'] and 
                        my_rect.centery <= intersection_bounds['bottom']):
                        
                        if (other_rect.centerx >= intersection_bounds['left'] and 
                            other_rect.centerx <= intersection_bounds['right'] and
                            other_rect.centery >= intersection_bounds['top'] and 
                            other_rect.centery <= intersection_bounds['bottom']):
                            
                            # Both in intersection - check collision
                            if my_rect.colliderect(other_rect):
                                return False
        
        return True
    
    def _move_straight_right(self):
        """Move straight to the right."""
        if self._can_move_forward():
            self.x += self.speed
    
    def _move_straight_down(self):
        """Move straight down."""
        if self._can_move_forward():
            self.y += self.speed
    
    def _move_straight_left(self):
        """Move straight to the left."""
        if self._can_move_forward():
            self.x -= self.speed
    
    def _move_straight_up(self):
        """Move straight up."""
        if self._can_move_forward():
            self.y -= self.speed
    
    def _handle_right_turn(self):
        """Handle right turn movement with improved collision detection."""
        mid_x = IntersectionLayout.MID_POINTS[self.direction]['x']
        
        # Before reaching turn point
        if self.crossed == 0 or self.x + self.current_image.get_rect().width < mid_x:
            if self._can_move_forward() and self._check_intersection_clearance():
                self.x += self.speed
        else:
            # At turn point - start turning
            if self.turned == 0:
                self._rotate_vehicle()
                self.x += 1.5  # Reduced speed during turn
                self.y += 1.3
                if self.rotate_angle >= 90:
                    self.turned = 1
                    self.rotate_angle = 90  # Clamp rotation
            else:
                # After turn - moving in new direction
                if self._can_move_in_turn() and self._check_intersection_clearance():
                   self.y += self.speed * 0.8  # Slightly slower after turn
    
    def _handle_down_turn(self):
        """Handle down turn movement with improved collision detection."""
        mid_y = IntersectionLayout.MID_POINTS[self.direction]['y']
        
        if self.crossed == 0 or self.y + self.current_image.get_rect().height < mid_y:
            if self._can_move_forward() and self._check_intersection_clearance():
                self.y += self.speed
        else:
            if self.turned == 0:
                self._rotate_vehicle()
                self.x -= 1.8  # Reduced speed during turn
                self.y += 1.5
                if self.rotate_angle >= 90:
                    self.turned = 1
                    self.rotate_angle = 90
            else:
                if self._can_move_in_turn() and self._check_intersection_clearance():
                    self.x -= self.speed * 0.8
    
    def _handle_left_turn(self):
        """Handle left turn movement with improved collision detection."""
        mid_x = IntersectionLayout.MID_POINTS[self.direction]['x']
        
        if self.crossed == 0 or self.x > mid_x:
            if self._can_move_forward() and self._check_intersection_clearance():
                self.x -= self.speed
        else:
            if self.turned == 0:
                self._rotate_vehicle()
                self.x -= 1.3
                self.y -= 1.8  # Reduced speed during turn
                if self.rotate_angle >= 90:
                    self.turned = 1
                    self.rotate_angle = 90
            else:
                if self._can_move_in_turn() and self._check_intersection_clearance():
                    self.y -= self.speed * 0.8
    
    def _handle_up_turn(self):
        """Handle up turn movement with improved collision detection."""
        mid_y = IntersectionLayout.MID_POINTS[self.direction]['y']
        
        if self.crossed == 0 or self.y > mid_y:
            if self._can_move_forward() and self._check_intersection_clearance():
                self.y -= self.speed
        else:
            if self.turned == 0:
                self._rotate_vehicle()
                self.x += 0.8
                self.y -= 0.8  # Reduced speed during turn
                if self.rotate_angle >= 90:
                    self.turned = 1
                    self.rotate_angle = 90
            else:
                if self._can_move_in_turn() and self._check_intersection_clearance():
                    self.x += self.speed * 0.8
    
    def _rotate_vehicle(self):
        """Rotate vehicle during turn."""
        self.rotate_angle += Config.ROTATION_ANGLE
        self.current_image = pygame.transform.rotate(self.original_image, -self.rotate_angle)
    
    def _can_move_in_turn(self) -> bool:
        """Check if vehicle can move while turning."""
        vehicles_in_lane = SimulationManager.vehicles[self.direction][self.lane]
        if self.index == 0:
            return True
        
        ahead_vehicle = vehicles_in_lane[self.index - 1]
        gap = Config.MOVING_GAP
        
        # Simplified collision check during turns
        return (abs(self.x - ahead_vehicle.x) > gap or 
                abs(self.y - ahead_vehicle.y) > gap)


# ============================================================================
# SIMULATION MANAGEMENT (Enhanced adaptivity from test3.py)
# ============================================================================

class SimulationManager:
    """Manages the overall simulation state and flow."""
    
    # Global simulation state
    signals: List[TrafficSignal] = []
    time_elapsed = 0
    current_green = 0
    next_green = 1
    current_yellow = 0
    cycle_counter = 0
    
    # Vehicle management
    vehicles = {
        'right': {0: [], 1: [], 2: [], 'crossed': 0},
        'down': {0: [], 1: [], 2: [], 'crossed': 0},
        'left': {0: [], 1: [], 2: [], 'crossed': 0},
        'up': {0: [], 1: [], 2: [], 'crossed': 0}
    }
    
    stops = {
        'right': [580, 580, 580],
        'down': [320, 320, 320],
        'left': [810, 810, 810],
        'up': [545, 545, 545]
    }
    
    # Analytics (Enhanced from test3.py)
    waiting_times = {0: [], 1: [], 2: [], 3: []}
    queue_history = {0: [], 1: [], 2: [], 3: []}
    throughput_history = []  # From test3.py
    efficiency_history = []  # From test3.py
    time_stamps = []
    last_served = {0: 0, 1: 0, 2: 0, 3: 0}
    results = {
        "served_counts": {0: 0, 1: 0, 2: 0, 3: 0},
        "fairness_used": 0,
        "adaptive_used": 0,
        "total_wait_time": 0,  # From test3.py
        "avg_throughput": 0,   # From test3.py
        "system_efficiency": 0,  # From test3.py
        "decision_log": []     # From test3.py
    }
    
    # Pygame components
    simulation = pygame.sprite.Group()

    trained_agents = None
    
    @classmethod
    def initialize(cls, training=False):
        """Initialize the simulation with default traffic signals."""
        cls.signals.clear()
        
        # Create four traffic signals
        ts1 = TrafficSignal(0, Config.DEFAULT_YELLOW, Config.DEFAULT_GREEN, 
                           Config.DEFAULT_MINIMUM, Config.DEFAULT_MAXIMUM)
        cls.signals.append(ts1)
        
        ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, Config.DEFAULT_YELLOW, 
                           Config.DEFAULT_GREEN, Config.DEFAULT_MINIMUM, Config.DEFAULT_MAXIMUM)
        cls.signals.append(ts2)
        
        ts3 = TrafficSignal(Config.DEFAULT_RED, Config.DEFAULT_YELLOW, Config.DEFAULT_GREEN,
                           Config.DEFAULT_MINIMUM, Config.DEFAULT_MAXIMUM)
        cls.signals.append(ts3)
        
        ts4 = TrafficSignal(Config.DEFAULT_RED, Config.DEFAULT_YELLOW, Config.DEFAULT_GREEN,
                           Config.DEFAULT_MINIMUM, Config.DEFAULT_MAXIMUM)
        cls.signals.append(ts4)
        
        # Start the signal control thread (use enhanced loop from test3.py)
        if not training:
            signal_thread = threading.Thread(target=cls._enhanced_signal_control_loop, daemon=True)
            signal_thread.start()
    
    @classmethod
    def calculate_demand_score(cls, direction_idx: int) -> float:  # From test3.py (replaces calculate_required_time)
        """Calculate demand score based on vehicle count, types, and wait times."""
        direction = Vehicle.DIRECTION_NUMBERS[direction_idx]
        vehicle_counts = {'car': 0, 'bus': 0, 'truck': 0, 'rickshaw': 0, 'bike': 0}
        total_wait_time = 0
        
        # Count vehicles and accumulated wait time
        for lane in range(3):
            for vehicle in cls.vehicles[direction][lane]:
                if vehicle.crossed == 0:
                    vehicle_counts[vehicle.vehicle_class] += 1
                    total_wait_time += vehicle.wait_time
        
        total_vehicles = sum(vehicle_counts.values())
        if total_vehicles == 0:
            return 0.0
        
        # Calculate base time requirement
        base_time = sum(count * Config.VEHICLE_TIMES[vtype] 
                       for vtype, count in vehicle_counts.items())
        green_time = math.ceil(base_time / Config.NUM_LANES)
        
        # Enhanced scoring with multiple factors
        demand_score = (
            green_time * 0.4 +                    # Base demand (40%)
            (total_wait_time / max(total_vehicles, 1)) * 0.3 +  # Wait time factor (30%)
            total_vehicles * 2 * 0.3              # Queue length factor (30%)
        )
        
        return max(0.0, demand_score)
    
    @classmethod
    def _enhanced_signal_control_loop(cls):  # From test3.py (replaces _signal_control_loop)
        """Main signal control loop with enhanced adaptive timing and fairness."""
        while True:
            # Green phase
            while cls.signals[cls.current_green].green > 0:
                cls._update_enhanced_timers()  # Enhanced timer from test3.py
                time.sleep(1)
            
            # Yellow phase
            cls.current_yellow = 1
            cls._reset_vehicle_stops()
            
            while cls.signals[cls.current_green].yellow > 0:
                cls._update_enhanced_timers()
                time.sleep(1)
            
            # End of cycle - choose next signal adaptively
            cls.current_yellow = 0
            cls._reset_current_signal()
            cls._enhanced_adaptive_selection()  # From test3.py
            
            # Update current signal
            cls.current_green = cls.next_green
            cls.next_green = (cls.current_green + 1) % Config.NUM_SIGNALS
            cls._update_red_times()
    
    @classmethod
    def _update_enhanced_timers(cls):  # From test3.py (replaces _update_signal_timers)
        """Update signal timers and record enhanced analytics."""
        cls.time_stamps.append(cls.time_elapsed)
        
        # Record queue lengths and enhanced metrics
        total_queue = 0
        total_wait = 0
        
        for direction_idx in range(Config.NUM_SIGNALS):
            direction = Vehicle.DIRECTION_NUMBERS[direction_idx]
            queue_count = sum(len([v for v in cls.vehicles[direction][lane] if v.crossed == 0])
                            for lane in range(3))
            cls.queue_history[direction_idx].append(queue_count)
            
            direction_wait = sum(v.wait_time for lane in range(3) for v in cls.vehicles[direction][lane] if v.crossed == 0)
            total_queue += queue_count
            total_wait += direction_wait
        
        # Calculate real-time metrics (from test3.py)
        if cls.time_elapsed > 0:
            current_throughput = sum(cls.vehicles[Vehicle.DIRECTION_NUMBERS[i]]['crossed'] 
                                   for i in range(4)) / cls.time_elapsed * 60
            cls.throughput_history.append(current_throughput)
            
            efficiency = max(0, 100 - (total_wait / max(total_queue, 1)))
            cls.efficiency_history.append(efficiency)
        
        # Update signal timers
        for i in range(Config.NUM_SIGNALS):
            if i == cls.current_green:
                if cls.current_yellow == 0:
                    cls.signals[i].green = max(0, cls.signals[i].green - 1)
                    cls.signals[i].total_green_time += 1
                else:
                    cls.signals[i].yellow = max(0, cls.signals[i].yellow - 1)
            else:
                cls.signals[i].red = max(0, cls.signals[i].red - 1)
    
    @classmethod
    def _reset_vehicle_stops(cls):
        """Reset vehicle stop positions for current direction."""
        direction = Vehicle.DIRECTION_NUMBERS[cls.current_green]
        for lane in range(3):
            cls.stops[direction][lane] = IntersectionLayout.DEFAULT_STOPS[direction]
            for vehicle in cls.vehicles[direction][lane]:
                vehicle.stop = IntersectionLayout.DEFAULT_STOPS[direction]
    
    @classmethod
    def _reset_current_signal(cls):
        """Reset current signal to default values."""
        cls.signals[cls.current_green].green = Config.DEFAULT_GREEN
        cls.signals[cls.current_green].yellow = Config.DEFAULT_YELLOW
        cls.signals[cls.current_green].red = Config.DEFAULT_RED
    
    @classmethod
    def _enhanced_adaptive_selection(cls):
        """Select next signal using MARL agents instead of heuristics."""
        cls.cycle_counter += 1

        # Get current observations (similar to TrafficMARLEnv._get_observations)
        obs = {}
        global_throughput = sum(cls.vehicles[dir]['crossed'] for dir in cls.vehicles) / max(cls.time_elapsed, 1)
        for i in range(Config.NUM_SIGNALS):
            dir = Vehicle.DIRECTION_NUMBERS[i]
            queues = [len([v for v in cls.vehicles[dir][lane] if v.crossed == 0]) for lane in range(3)]
            avg_wait = np.mean(cls.waiting_times[i]) if cls.waiting_times[i] else 0
            phase = [1 if cls.signals[i].green > 0 else 0,  # Green
                     1 if cls.signals[i].yellow > 0 else 0,  # Yellow
                     1 if cls.signals[i].red > 0 else 0]     # Red
            obs[i] = np.array(queues + [avg_wait] + phase + [global_throughput], dtype=np.float32)

        # Query agents for actions (no exploration in inference)
        actions = {}
        for i in range(Config.NUM_SIGNALS):
            state = torch.FloatTensor(obs[i])
            q_values = cls.trained_agents[i](state)  # Use trained_agents
            actions[i] = torch.argmax(q_values).item()

        # Apply the action with highest confidence or cycle through (for multi-agent coord)
        selected_direction = max(actions, key=lambda k: actions[k])  # Simple: pick direction with max action value
        green_time = (actions[selected_direction] + 1) * 10  # Map to 10-50s

        # Activate the signal
        cls.next_green = selected_direction
        cls.signals[cls.next_green].activate(green_time)

        # Update tracking (keep some original logic)
        cls.last_served[cls.next_green] = cls.cycle_counter
        cls.results['served_counts'][cls.next_green] = cls.vehicles[
            Vehicle.DIRECTION_NUMBERS[cls.next_green]]['crossed']

        cls.results["decision_log"].append({"cycle": cls.cycle_counter, "type": "MARL"})

        cls._log_selection_decision()
    
    @classmethod
    def _get_fairness_candidates(cls) -> Dict[int, float]:
        """Get directions that require fairness enforcement."""
        candidates = {}
        for direction in range(Config.NUM_SIGNALS):
            cycles_waiting = cls.cycle_counter - cls.last_served[direction]
            if cycles_waiting >= Config.FAIRNESS_LIMITS[direction]:
                demand = cls.calculate_demand_score(direction)
                if demand > 0:
                    candidates[direction] = demand * (cycles_waiting / Config.FAIRNESS_LIMITS[direction])
        return candidates
    
    @classmethod
    def _get_adaptive_candidates(cls) -> Dict[int, float]:
        """Get directions for adaptive selection based on demand."""
        candidates = {}
        for direction in range(Config.NUM_SIGNALS):
            demand = cls.calculate_demand_score(direction)
            if demand > 0:
                candidates[direction] = demand
        return candidates
    
    @classmethod
    def _update_red_times(cls):
        """Update red times for other signals."""
        cls.signals[cls.next_green].red = (cls.signals[cls.current_green].yellow + 
                                          cls.signals[cls.current_green].green)
    
    @classmethod
    def _can_spawn_vehicles(cls) -> bool:
        """Check if it's safe to spawn new vehicles to prevent overcrowding."""
        total_vehicles = 0
        for direction in ['right', 'down', 'left', 'up']:
            for lane in range(3):
                total_vehicles += len([v for v in cls.vehicles[direction][lane] if v.crossed == 0])
        
        # Limit total vehicles in the system to prevent gridlock
        return total_vehicles < 60  # Reduced from potential unlimited spawning
    
    @classmethod
    def _log_selection_decision(cls):
        """Log the signal selection decision."""
        direction_name = Vehicle.DIRECTION_NUMBERS[cls.next_green].upper()
        green_time = cls.signals[cls.next_green].green
        selection_type = "FAIRNESS" if cls.results['fairness_used'] > cls.results['adaptive_used'] else "ADAPTIVE"
        
        demands = [cls.calculate_demand_score(d) for d in range(Config.NUM_SIGNALS)]
        
        print(f"[Cycle {cls.cycle_counter}] {selection_type} -> {direction_name} "
              f"gets GREEN for {green_time}s")
        print(f"[Cycle {cls.cycle_counter}] demands (R,D,L,U) = {demands}")


# ============================================================================
# VEHICLE GENERATION
# ============================================================================

class VehicleGenerator:
    """Handles vehicle generation for the simulation."""
    
    @staticmethod
    def start_generation():
        """Start vehicle generation in a separate thread."""
        generation_thread = threading.Thread(target=VehicleGenerator._generation_loop, daemon=True)
        generation_thread.start()
    
    @staticmethod
    def _generation_loop():
        """Main vehicle generation loop with controlled spawning."""
        while True:
            VehicleGenerator.generate_vehicles()
            time.sleep(1.0)  # Slower generation rate to prevent overcrowding
    
    @staticmethod
    def generate_vehicles():
        """Generate vehicles for one time step."""
        # Check if there's space to spawn new vehicles
        if SimulationManager._can_spawn_vehicles():
            vehicle_type = random.randint(0, 4)
            
            # Bikes use rightmost lane, others use center lanes
            if vehicle_type == 4:  # bike
                lane_number = 0
            else:
                lane_number = random.randint(1, 2)
            
            # Determine turning behavior (reduced turn probability to reduce intersection conflicts)
            will_turn = 0
            if lane_number == 2:  # rightmost lane
                will_turn = 1 if random.randint(0, 6) <= 2 else 0  # Reduced from 4 to 6
            
            # Select direction with equal probability
            direction_number = random.choices([0, 1, 2, 3], weights=[0.25, 0.25, 0.25, 0.25])[0]
            
            # Create vehicle
            Vehicle(lane_number, 
                   Vehicle.VEHICLE_TYPES[vehicle_type],
                   direction_number,
                   Vehicle.DIRECTION_NUMBERS[direction_number],
                   will_turn)


# ============================================================================
# ANALYTICS AND REPORTING (Minor enhancements from test3.py)
# ============================================================================

class AnalyticsManager:
    """Handles simulation analytics and reporting."""
    
    @staticmethod
    def start_time_tracking():
        """Start simulation time tracking thread."""
        time_thread = threading.Thread(target=AnalyticsManager._time_tracking_loop, daemon=True)
        time_thread.start()
    
    @staticmethod
    def _time_tracking_loop():
        """Main time tracking loop."""
        while True:
            SimulationManager.time_elapsed += 1
            time.sleep(1)
            
            if SimulationManager.time_elapsed >= Config.SIMULATION_TIME:
                AnalyticsManager.generate_final_report()
                os._exit(1)
    
    @staticmethod
    def generate_final_report():
        """Generate and display final simulation report with analytics."""
        print('\n' + '='*50)
        print('SIMULATION SUMMARY')
        print('='*50)
        
        total_vehicles = 0
        print('\nLane-wise Vehicle Counts and Average Wait Times:')
        print('-' * 50)
        
        for i in range(Config.NUM_SIGNALS):
            direction_name = Vehicle.DIRECTION_NUMBERS[i]
            served = SimulationManager.vehicles[direction_name]['crossed']
            total_vehicles += served
            
            avg_wait = (sum(SimulationManager.waiting_times[i]) / 
                       len(SimulationManager.waiting_times[i])) if SimulationManager.waiting_times[i] else 0.0
            
            print(f"Lane {i+1} ({direction_name.capitalize()}): "
                  f"Served={served} | Avg Wait={avg_wait:.2f}s")
            SimulationManager.results['served_counts'][i] = served
        
        print(f'\nTotal vehicles served: {total_vehicles}')
        print(f'Total simulation time: {SimulationManager.time_elapsed}s')
        
        # Calculate overall statistics (enhanced from test3.py)
        all_waiting_times = []
        for direction_times in SimulationManager.waiting_times.values():
            all_waiting_times.extend(direction_times)
        
        overall_avg_wait = (sum(all_waiting_times) / len(all_waiting_times)) if all_waiting_times else 0.0
        throughput = (total_vehicles / SimulationManager.time_elapsed) * 60.0 if SimulationManager.time_elapsed > 0 else 0.0
        SimulationManager.results['avg_throughput'] = throughput
        SimulationManager.results['system_efficiency'] = max(0, 100 - overall_avg_wait) if overall_avg_wait > 0 else 100
        
        print(f'Overall average wait time: {overall_avg_wait:.2f}s')
        print(f'Throughput: {throughput:.2f} vehicles/minute')
        
        print(f'\nAdaptive decisions: {SimulationManager.results["adaptive_used"]}')
        print(f'Fairness enforcements: {SimulationManager.results["fairness_used"]}')
        
        # Generate plots
        AnalyticsManager._generate_plots()
    
    @staticmethod
    def _generate_plots():
        """Generate all analytical plots in a single side-by-side dashboard. (Simplified from test3.py)"""
        try:
            directions = [Vehicle.DIRECTION_NUMBERS[i].capitalize() for i in range(Config.NUM_SIGNALS)]
            served_counts = [SimulationManager.results['served_counts'][i] for i in range(Config.NUM_SIGNALS)]
            avg_waits = [(sum(SimulationManager.waiting_times[i]) / len(SimulationManager.waiting_times[i])) 
                        if SimulationManager.waiting_times[i] else 0.0 for i in range(Config.NUM_SIGNALS)]
            
            # Create a comprehensive dashboard with all plots
            fig = plt.figure(figsize=(20, 12))
            fig.suptitle('Traffic Signal Simulation - Complete Analytics Dashboard', 
                        fontsize=20, fontweight='bold', y=0.98)
            
            # Plot 1: Vehicles Served per Lane (Top Left)
            ax1 = plt.subplot(2, 3, 1)
            bars1 = ax1.bar(directions, served_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], 
                           edgecolor='black', linewidth=1)
            ax1.set_title('Vehicles Served per Lane', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Direction', fontsize=11)
            ax1.set_ylabel('Vehicles Served', fontsize=11)
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Plot 2: Average Waiting Time per Lane (Top Center)
            ax2 = plt.subplot(2, 3, 2)
            bars2 = ax2.bar(directions, avg_waits, color=['#FFB74D', '#81C784', '#64B5F6', '#F06292'], 
                           edgecolor='black', linewidth=1)
            ax2.set_title('Average Waiting Time per Lane', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Direction', fontsize=11)
            ax2.set_ylabel('Avg Wait Time (s)', fontsize=11)
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Plot 3: Algorithm Usage Distribution (Top Right)
            ax3 = plt.subplot(2, 3, 3)
            if SimulationManager.results['adaptive_used'] + SimulationManager.results['fairness_used'] > 0:
                labels = ['Adaptive\nDecisions', 'Fairness\nEnforcements']
                sizes = [SimulationManager.results['adaptive_used'], SimulationManager.results['fairness_used']]
                colors = ['#3498DB', '#E74C3C']
                explode = (0.05, 0.05)
                
                wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels, colors=colors, 
                                                  autopct='%1.1f%%', shadow=True, startangle=90, 
                                                  textprops={'fontsize': 10})
                ax3.set_title('Algorithm Usage Distribution', fontsize=14, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'No Algorithm\nData Available', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Algorithm Usage Distribution', fontsize=14, fontweight='bold')
            
            # Plot 4: Queue Length Evolution Over Time (Bottom Span)
            ax4 = plt.subplot(2, 1, 2)
            colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
            
            for direction_idx in range(Config.NUM_SIGNALS):
                direction_name = Vehicle.DIRECTION_NUMBERS[direction_idx].capitalize()
                ax4.plot(SimulationManager.time_stamps, 
                        SimulationManager.queue_history[direction_idx], 
                        label=direction_name, 
                        color=colors[direction_idx], 
                        linewidth=2.5,
                        marker='o' if len(SimulationManager.time_stamps) < 50 else None,
                        markersize=4)
            
            ax4.set_title('Queue Length Evolution Over Time', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Simulation Time (seconds)', fontsize=11)
            ax4.set_ylabel('Queue Length (vehicles)', fontsize=11)
            ax4.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
            ax4.grid(True, linestyle='--', alpha=0.6)
            
            # Add performance summary text box
            total_vehicles = sum(served_counts)
            overall_waits = []
            for d in range(4): 
                overall_waits.extend(SimulationManager.waiting_times[d])
            overall_avg_wait = (sum(overall_waits)/len(overall_waits)) if overall_waits else 0.0
            throughput = (total_vehicles / SimulationManager.time_elapsed) * 60.0 if SimulationManager.time_elapsed > 0 else 0.0
            
            summary_text = f"""SIMULATION SUMMARY
Total Vehicles: {total_vehicles}
Simulation Time: {SimulationManager.time_elapsed}s
Overall Avg Wait: {overall_avg_wait:.1f}s
Throughput: {throughput:.1f} veh/min
Cycles Completed: {SimulationManager.cycle_counter}"""
            
            # Add text box in the empty space
            fig.text(0.02, 0.35, summary_text, fontsize=11, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                    verticalalignment='top')
            
            # Adjust layout to prevent overlap
            plt.tight_layout(rect=[0.15, 0.03, 0.95, 0.95])
            
            # Show the comprehensive dashboard
            plt.show()
                
        except Exception as e:
            print(f'Error generating plots: {e}')


# ============================================================================
# VISUALIZATION AND UI (Enhanced from test3.py)
# ============================================================================

class VisualizationManager:
    """Manages the Pygame-based visualization."""
    
    # Enhanced color palette (from test3.py)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (231, 76, 60)      # Modern red
    YELLOW = (241, 196, 15)  # Modern yellow  
    GREEN = (46, 204, 113)   # Modern green
    BLUE = (52, 152, 219)    # Modern blue
    GRAY = (149, 165, 166)   # Modern gray
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        pygame.display.set_caption("Adaptive Traffic Signal Simulation")
        
        # Load resources
        self._load_images()
        self._load_fonts()
        
        # Display elements
        self.vehicle_count_texts = ["0", "0", "0", "0"]
        self.clock = pygame.time.Clock()
    
    def _load_images(self):
        """Load background and signal images with fallbacks. (From test3.py)"""
        try:
            self.background = pygame.image.load('images/mod_int.png')
        except Exception:
            # Enhanced fallback background
            self.background = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
            self.background.fill((45, 52, 54))  # Dark modern background
            
            # Add intersection lines
            pygame.draw.rect(self.background, (100, 100, 100), (590, 0, 220, Config.SCREEN_HEIGHT))
            pygame.draw.rect(self.background, (100, 100, 100), (0, 330, Config.SCREEN_WIDTH, 205))
        
        try:
            self.red_signal = pygame.image.load('images/signals/red.png')
            self.yellow_signal = pygame.image.load('images/signals/yellow.png')
            self.green_signal = pygame.image.load('images/signals/green.png')
        except Exception:
            self.red_signal = self.yellow_signal = self.green_signal = None
    
    def _load_fonts(self):
        """Load enhanced fonts. (From test3.py)"""
        try:
            self.font = pygame.font.Font(None, 28)
            self.large_font = pygame.font.Font(None, 34)
            self.small_font = pygame.font.Font(None, 22)
        except:
            self.font = self.large_font = self.small_font = pygame.font.Font(None, 24)
    
    def render_frame(self):
        """Render a single frame of the simulation. (Enhanced from test3.py)"""
        self.screen.blit(self.background, (0, 0))
        
        self._render_enhanced_signals()  # From test3.py
        self._render_vehicles()
        self._render_enhanced_ui()  # From test3.py
        
        pygame.display.update()
        self.clock.tick(Config.FPS)
    
    def _render_enhanced_signals(self):
        """Render enhanced traffic signals with modern styling. (From test3.py)"""
        for i in range(Config.NUM_SIGNALS):
            signal = SimulationManager.signals[i]
            coord = IntersectionLayout.SIGNAL_COORDS[i]
            
            # Enhanced signal display with background
            bg_rect = pygame.Rect(coord[0]-5, coord[1]-5, 50, 50)
            pygame.draw.rect(self.screen, self.BLACK, bg_rect)
            pygame.draw.rect(self.screen, self.WHITE, bg_rect, 2)
            
            if i == SimulationManager.current_green:
                if SimulationManager.current_yellow == 1:
                    self._draw_enhanced_signal(coord, self.YELLOW, "YELLOW")
                    signal.signal_text = str(signal.yellow) if signal.yellow > 0 else "STOP"
                else:
                    self._draw_enhanced_signal(coord, self.GREEN, "GREEN")
                    signal.signal_text = str(signal.green) if signal.green > 0 else "GO"
            else:
                self._draw_enhanced_signal(coord, self.RED, "RED")
                signal.signal_text = str(signal.red) if signal.red <= 10 else "---"
            
            # Enhanced timer text
            timer_coord = IntersectionLayout.SIGNAL_TIMER_COORDS[i]
            timer_bg = pygame.Rect(timer_coord[0]-2, timer_coord[1]-2, 44, 24)
            pygame.draw.rect(self.screen, self.BLACK, timer_bg)
            timer_text = self.font.render(str(signal.signal_text), True, self.WHITE)
            self.screen.blit(timer_text, timer_coord)
    
    def _draw_enhanced_signal(self, coord: Tuple[int, int], color: Tuple[int, int, int], state: str):
        """Draw enhanced signal light with glow effect. (From test3.py)"""
        center = (coord[0] + 20, coord[1] + 20)
        
        # Outer glow
        for radius in range(25, 15, -2):
            alpha = max(0, 50 - (25 - radius) * 10)
            glow_color = (*color, alpha)
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color, (radius, radius), radius)
            self.screen.blit(s, (center[0] - radius, center[1] - radius))
        
        # Main signal
        pygame.draw.circle(self.screen, color, center, 15)
        pygame.draw.circle(self.screen, self.WHITE, center, 15, 2)
    
    def _render_vehicles(self):
        """Render all vehicles in the simulation."""
        for vehicle in SimulationManager.simulation:
            try:
                self.screen.blit(vehicle.current_image, (vehicle.x, vehicle.y))
                vehicle.move()
            except Exception:
                # Handle any rendering errors gracefully
                pass
    
    def _render_enhanced_ui(self):
        """Render enhanced user interface with modern styling. (From test3.py)"""
        # Enhanced vehicle count displays
        for i in range(Config.NUM_SIGNALS):
            direction_name = Vehicle.DIRECTION_NUMBERS[i]
            crossed_count = SimulationManager.vehicles[direction_name]['crossed']
            
            count_coord = IntersectionLayout.VEHICLE_COUNT_COORDS[i]
            
            # Background for count
            count_bg = pygame.Rect(count_coord[0]-2, count_coord[1]-2, 36, 24)
            pygame.draw.rect(self.screen, self.BLACK, count_bg)
            pygame.draw.rect(self.screen, self.BLUE, count_bg, 2)
            
            count_text = self.font.render(str(crossed_count), True, self.WHITE)
            self.screen.blit(count_text, count_coord)
        
        # Enhanced info panel
        info_x, info_y = 1050, 30
        panel_width, panel_height = 320, 200
        
        # Info panel background
        info_panel = pygame.Rect(info_x, info_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, (40, 44, 52), info_panel)
        pygame.draw.rect(self.screen, self.BLUE, info_panel, 3)
        
        # Info text with enhanced formatting
        info_texts = [
            f"🕒 Time: {SimulationManager.time_elapsed}s",
            f"🔄 Cycle: {SimulationManager.cycle_counter}",
            f"🚦 Active: {Vehicle.DIRECTION_NUMBERS[SimulationManager.current_green].upper()}",
            f"📊 Phase: {'YELLOW' if SimulationManager.current_yellow else 'GREEN'}",
            f"🧠 Adaptive: {SimulationManager.results['adaptive_used']}",
            f"⚖️ Fairness: {SimulationManager.results['fairness_used']}",
            "",
            "Controls:",
            "SPACE - Status",
            "R - Analytics", 
            "ESC - Exit"
        ]
        
        for i, text in enumerate(info_texts):
            if text.startswith("Controls:"):
                color = self.YELLOW
                font = self.small_font
            elif text in ["SPACE - Status", "R - Analytics", "ESC - Exit"]:
                color = self.GRAY
                font = self.small_font
            else:
                color = self.WHITE
                font = self.small_font if text == "" else self.font
            
            if text:  # Skip empty lines
                rendered_text = font.render(text, True, color)
                self.screen.blit(rendered_text, (info_x + 10, info_y + 10 + i * 16))
        
        # Enhanced throughput display (from test3.py)
        if SimulationManager.throughput_history:
            current_throughput = SimulationManager.throughput_history[-1]
            throughput_text = f"📈 Throughput: {current_throughput:.1f} veh/min"
            throughput_rendered = self.font.render(throughput_text, True, self.GREEN)
            self.screen.blit(throughput_rendered, (info_x + 10, info_y + panel_height + 10))


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class TrafficMARLEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_agents = Config.NUM_SIGNALS  # 4 agents
        # State: per agent [queue_lengths (3), avg_wait, phase_onehot (3), global_throughput]
        self.observation_space = {i: Box(low=0, high=np.inf, shape=(3 + 1 + 3 + 1,)) for i in range(self.num_agents)}
        # Action: discrete green time bucket per agent (5 choices: 10-50s in 10s steps)
        self.action_space = {i: Discrete(5) for i in range(self.num_agents)}
        self.current_step = 0
        self.max_steps = 100  # Episode length, in cycles
    
    def reset(self, seed=None):
        # Reset sim: clear vehicles, reset signals, time=0
        SimulationManager.initialize(training=True)
        SimulationManager.time_elapsed = 0
        SimulationManager.vehicles = {
            'right': {0: [], 1: [], 2: [], 'crossed': 0},
            'down': {0: [], 1: [], 2: [], 'crossed': 0},
            'left': {0: [], 1: [], 2: [], 'crossed': 0},
            'up': {0: [], 1: [], 2: [], 'crossed': 0}
        }
        SimulationManager.stops = {
            'right': [580, 580, 580],
            'down': [320, 320, 320],
            'left': [810, 810, 810],
            'up': [545, 545, 545]
        }
        SimulationManager.simulation = pygame.sprite.Group()
        IntersectionLayout.START_X = {
            'right': [0, 0, 0],
            'down': [755, 727, 697],
            'left': [1400, 1400, 1400],
            'up': [602, 627, 657]
        }
        IntersectionLayout.START_Y = {
            'right': [348, 370, 398],
            'down': [0, 0, 0],
            'left': [498, 466, 436],
            'up': [800, 830, 860]
        }
        # Reset other states as needed (e.g., clear waiting_times, queue_history, etc.)
        SimulationManager.waiting_times = {0: [], 1: [], 2: [], 3: []}
        SimulationManager.queue_history = {0: [], 1: [], 2: [], 3: []}
        SimulationManager.throughput_history = []
        SimulationManager.efficiency_history = []
        SimulationManager.time_stamps = []
        SimulationManager.results = {
            "served_counts": {0: 0, 1: 0, 2: 0, 3: 0},
            "fairness_used": 0,
            "adaptive_used": 0,
            "total_wait_time": 0,
            "avg_throughput": 0,
            "system_efficiency": 0,
            "decision_log": []
        }
        observations = self._get_observations()
        self.current_step = 0
        return observations, {}  # For multi-agent dict format
    
    def step(self, actions):
        # Select direction with max action value
        selected_direction = max(actions, key=lambda k: actions[k])
        green_time = (actions[selected_direction] + 1) * 10

        # Activate the signal
        SimulationManager.next_green = selected_direction
        SimulationManager.signals[selected_direction].activate(green_time)
        SimulationManager.current_green = selected_direction
        SimulationManager.next_green = (SimulationManager.current_green + 1) % Config.NUM_SIGNALS
        SimulationManager._update_red_times()
        SimulationManager.cycle_counter += 1
        SimulationManager.results["decision_log"].append({"cycle": SimulationManager.cycle_counter, "type": "MARL"})

        # Simulate the green phase
        for _ in range(green_time):
            for vehicle in SimulationManager.simulation.sprites():
                vehicle.move()
            SimulationManager._update_enhanced_timers()
            VehicleGenerator.generate_vehicles()
            SimulationManager.time_elapsed += 1
        
        # Yellow phase
        SimulationManager.current_yellow = 1
        SimulationManager._reset_vehicle_stops()
        for _ in range(Config.DEFAULT_YELLOW):
            for vehicle in SimulationManager.simulation.sprites():
                vehicle.move()
            SimulationManager._update_enhanced_timers()
            VehicleGenerator.generate_vehicles()
            SimulationManager.time_elapsed += 1
        
        SimulationManager.current_yellow = 0
        SimulationManager._reset_current_signal()

        observations = self._get_observations()
        rewards = self._get_rewards()
        terminated = {i: self.current_step >= self.max_steps for i in range(self.num_agents)}
        truncated = terminated.copy()  # Or custom
        self.current_step += 1
        return observations, rewards, terminated, truncated, {}
    
    def _get_observations(self):
        obs = {}
        global_throughput = sum(SimulationManager.vehicles[dir]['crossed'] for dir in SimulationManager.vehicles) / max(SimulationManager.time_elapsed, 1)
        for i in range(self.num_agents):
            dir = Vehicle.DIRECTION_NUMBERS[i]
            queues = [len([v for v in SimulationManager.vehicles[dir][lane] if v.crossed == 0]) for lane in range(3)]
            avg_wait = np.mean(SimulationManager.waiting_times[i]) if SimulationManager.waiting_times[i] else 0
            phase = [1 if SimulationManager.signals[i].green > 0 else 0,  # Green
                     1 if SimulationManager.signals[i].yellow > 0 else 0,  # Yellow
                     1 if SimulationManager.signals[i].red > 0 else 0]     # Red
            obs[i] = np.array(queues + [avg_wait] + phase + [global_throughput], dtype=np.float32)
        return obs
    
    def _get_rewards(self):
        rewards = {}
        for i in range(self.num_agents):
            # Per-agent reward: -avg wait in direction + local throughput
            local_wait = np.mean(SimulationManager.waiting_times[i]) if SimulationManager.waiting_times[i] else 0
            local_throughput = SimulationManager.vehicles[Vehicle.DIRECTION_NUMBERS[i]]['crossed'] / max(SimulationManager.time_elapsed, 1)
            rewards[i] = -local_wait + local_throughput
        return rewards


# Training setup
state_size = 3 + 1 + 3 + 1  # From obs shape: queues(3) + avg_wait(1) + phase(3) + global_throughput(1)
action_size = 5
def train_marl(num_episodes=50, batch_size=32):
    env = TrafficMARLEnv()
    agents = [DQN(state_size, action_size) for _ in range(Config.NUM_SIGNALS)]
    optimizers = [optim.Adam(agent.parameters(), lr=0.001) for agent in agents]
    replay_buffers = [deque(maxlen=10000) for _ in range(Config.NUM_SIGNALS)]  # Per agent experience
    epsilon = 1.0  # Exploration
    gamma = 0.99  # Discount
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = {i: 0 for i in range(env.num_agents)}
        done = False
        while not done:
            actions = {}
            for i in range(env.num_agents):
                state = torch.FloatTensor(obs[i])
                if np.random.rand() < epsilon:
                    actions[i] = np.random.randint(0, action_size)
                else:
                    q_values = agents[i](state)
                    actions[i] = torch.argmax(q_values).item()
            
            next_obs, rewards, term, trunc, _ = env.step(actions)
            done = all(term.values())
            
            # Store and train per agent
            for i in range(env.num_agents):
                replay_buffers[i].append((obs[i], actions[i], rewards[i], next_obs[i], term[i]))
                total_reward[i] += rewards[i]
                
                if len(replay_buffers[i]) > batch_size:
                    batch = random.sample(replay_buffers[i], batch_size)
                    states, acts, rews, next_states, dones = zip(*batch)
                    states = torch.FloatTensor(np.array(states))
                    next_states = torch.FloatTensor(np.array(next_states))
                    q_values = agents[i](states).gather(1, torch.LongTensor(acts).unsqueeze(1)).squeeze(1)
                    next_q = agents[i](next_states).max(1)[0]
                    targets = torch.FloatTensor(rews) + gamma * next_q * (1 - torch.FloatTensor(dones))
                    loss = nn.MSELoss()(q_values, targets.detach())
                    optimizers[i].zero_grad()
                    loss.backward()
                    optimizers[i].step()
        
        epsilon = max(0.01, epsilon * 0.995)  # Decay
        print(f"Episode {episode}: Rewards {total_reward}")
    
    # Save trained models (optional)
    for i, agent in enumerate(agents):
        torch.save(agent.state_dict(), f"agent_{i}.pth")
    
    return agents  # Return for inference


# ============================================================================
# MAIN APPLICATION (Enhanced key handlers and prints from test3.py)
# ============================================================================

class TrafficSimulationApp:
    """Main application class that orchestrates the entire simulation."""
    
    def __init__(self):
        self.visualization = VisualizationManager()
        self.running = True
        self.paused = False  # From test3.py
    
    def start_simulation(self):
        """Initialize and start all simulation components."""
        print("Initializing Traffic Signal Simulation...")
        print(f"Simulation Duration: {Config.SIMULATION_TIME} seconds")
        print(f"Number of Signals: {Config.NUM_SIGNALS}")
        print(f"Fairness Limits: {Config.FAIRNESS_LIMITS}")
        print("-" * 50)
        
        # Initialize simulation components
        SimulationManager.initialize()
        VehicleGenerator.start_generation()
        AnalyticsManager.start_time_tracking()
        
        print("Simulation started successfully!")
    
    async def run_main_loop(self):
        """Main simulation loop with Pygame event handling. (Enhanced from test3.py)"""
        await self.start_simulation_async()
        
        while self.running:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    await self._handle_keypress(event.key)  # Enhanced handler from test3.py
            
            # Render frame (skip if paused, from test3.py)
            if not self.paused:
                self.visualization.render_frame()
            
            # Async yield for smooth animation
            await asyncio.sleep(1.0 / Config.FPS)
    
    async def start_simulation_async(self):
        """Async wrapper for simulation initialization."""
        self.start_simulation()
    
    async def _handle_keypress(self, key):  # From test3.py
        """Enhanced keypress handling with new features."""
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            self._print_enhanced_status()  # From test3.py
        elif key == pygame.K_r:
            self._print_detailed_analytics()  # From test3.py
        elif key == pygame.K_p:
            self.paused = not self.paused
            print(f"⏸️ Simulation {'PAUSED' if self.paused else 'RESUMED'}")
    
    def _print_enhanced_status(self):  # From test3.py
        """Print enhanced current simulation status."""
        print(f"\n🔍 ENHANCED STATUS REPORT (t={SimulationManager.time_elapsed}s)")
        print("=" * 60)
        print(f"🔄 Cycle: {SimulationManager.cycle_counter}")
        print(f"🚦 Active Signal: {Vehicle.DIRECTION_NUMBERS[SimulationManager.current_green].upper()} "
              f"({'YELLOW' if SimulationManager.current_yellow else 'GREEN'})")
        
        print(f"\n📊 Direction Status:")
        total_queue = 0
        total_served = 0
        
        for i in range(Config.NUM_SIGNALS):
            direction = Vehicle.DIRECTION_NUMBERS[i]
            queue_size = sum(len([v for v in SimulationManager.vehicles[direction][lane] if v.crossed == 0])
                           for lane in range(3))
            served = SimulationManager.vehicles[direction]['crossed']
            signal = SimulationManager.signals[i]
            
            total_queue += queue_size
            total_served += served
            
            efficiency = (served / signal.total_green_time * 60) if signal.total_green_time > 0 else 0
            print(f"   {direction.upper():>5}: Queue={queue_size:>2} | Served={served:>3} | "
                  f"Activations={signal.activation_count:>2} | Eff={efficiency:>5.1f} v/min")
        
        print(f"\n🎯 System Metrics:")
        if SimulationManager.time_elapsed > 0:
            throughput = (total_served / SimulationManager.time_elapsed) * 60
            print(f"   • Current Throughput: {throughput:.2f} vehicles/minute")
        print(f"   • Total Queue Length: {total_queue}")
        print(f"   • Algorithm Ratio: {SimulationManager.results['adaptive_used']}A / {SimulationManager.results['fairness_used']}F")
    
    def _print_detailed_analytics(self):  # From test3.py
        """Print detailed analytics summary."""
        print(f"\n📈 DETAILED ANALYTICS (t={SimulationManager.time_elapsed}s)")
        print("=" * 65)
        
        # Calculate advanced metrics
        total_served = sum(SimulationManager.results['served_counts'].values())
        all_waits = []
        for waits in SimulationManager.waiting_times.values():
            all_waits.extend(waits)
        
        if all_waits:
            avg_wait = sum(all_waits) / len(all_waits)
            max_wait = max(all_waits)
            min_wait = min(all_waits)
            
            print(f"⏱️  Wait Time Analysis:")
            print(f"   • Average: {avg_wait:.1f}s")
            print(f"   • Range: {min_wait:.1f}s - {max_wait:.1f}s")
            print(f"   • Total samples: {len(all_waits)}")
        
        if SimulationManager.throughput_history:
            avg_throughput = sum(SimulationManager.throughput_history) / len(SimulationManager.throughput_history)
            max_throughput = max(SimulationManager.throughput_history)
            print(f"\n🚀 Throughput Analysis:")
            print(f"   • Average: {avg_throughput:.1f} vehicles/minute")
            print(f"   • Peak: {max_throughput:.1f} vehicles/minute")
        
        print(f"\n🤖 Algorithm Performance:")
        total_decisions = SimulationManager.results['adaptive_used'] + SimulationManager.results['fairness_used']
        if total_decisions > 0:
            adaptive_ratio = SimulationManager.results['adaptive_used'] / total_decisions
            print(f"   • Adaptive Preference: {adaptive_ratio:.1%}")
            print(f"   • Decision Efficiency: {total_served / total_decisions:.1f} vehicles/decision")


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

async def main():
    """Main entry point for the traffic simulation application."""
    try:
        app = TrafficSimulationApp()
        await app.run_main_loop()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        print("Simulation ended.")


# ============================================================================
# ADDITIONAL UTILITIES
# ============================================================================

def print_simulation_info():
    """Print detailed simulation configuration information."""
    print("\n" + "="*60)
    print("ADAPTIVE TRAFFIC SIGNAL SIMULATION")
    print("="*60)
    print(f"Configuration:")
    print(f"  • Signal Count: {Config.NUM_SIGNALS}")
    print(f"  • Lanes per Direction: {Config.NUM_LANES}")
    print(f"  • Default Green Time: {Config.DEFAULT_GREEN}s")
    print(f"  • Yellow Time: {Config.DEFAULT_YELLOW}s")
    print(f"  • Minimum Green: {Config.DEFAULT_MINIMUM}s")
    print(f"  • Maximum Green: {Config.DEFAULT_MAXIMUM}s")
    print(f"  • Simulation Duration: {Config.SIMULATION_TIME}s")
    print(f"\nFairness Limits:")
    for direction, limit in Config.FAIRNESS_LIMITS.items():
        direction_name = Vehicle.DIRECTION_NUMBERS[direction].capitalize()
        print(f"  • {direction_name}: {limit} cycles")
    print(f"\nVehicle Processing Times:")
    for vehicle_type, time_val in Config.VEHICLE_TIMES.items():
        print(f"  • {vehicle_type.capitalize()}: {time_val}s")
    print("\nControls:")
    print("  • SPACE: Print current status")
    print("  • R: Print analytics summary")
    print("  • ESC: Exit simulation")
    print("="*60 + "\n")


if __name__ == "__main__":
    # trained_agents = train_marl(num_episodes=200)
    # SimulationManager.trained_agents = trained_agents
    # print_simulation_info()
    if platform.system() == "Emscripten":
        asyncio.ensure_future(main())
    else:
        asyncio.run(main())