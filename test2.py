"""
Traffic Signal Simulation System
================================

An adaptive traffic signal control system with fairness mechanisms.
Features:
- Adaptive signal timing based on vehicle demand
- Fairness enforcement to prevent lane starvation
- Real-time visualization using Pygame
- Performance analytics and plotting

"""

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

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class Config:
    """Global configuration settings for the simulation."""
    
    # Signal timing defaults (seconds)
    DEFAULT_RED = 150
    DEFAULT_YELLOW = 5
    DEFAULT_GREEN = 30
    DEFAULT_MINIMUM = 10
    DEFAULT_MAXIMUM = 60
    
    # Simulation parameters
    NUM_SIGNALS = 4
    NUM_LANES = 3
    SIMULATION_TIME = 100
    
    # Vehicle parameters
    VEHICLE_TIMES = {
        'car': 2.0,
        'bike': 1.0,
        'rickshaw': 2.25,
        'bus': 2.5,
        'truck': 2.5
    }
    
    VEHICLE_SPEEDS = {
        'car': 2.25,
        'bus': 1.8,
        'truck': 1.8,
        'rickshaw': 2.0,
        'bike': 2.5
    }
    
    # Fairness limits (max cycles a lane can be skipped)
    FAIRNESS_LIMITS = {0: 5, 1: 3, 2: 4, 3: 3}
    
    # Display settings
    SCREEN_WIDTH = 1400
    SCREEN_HEIGHT = 820
    FPS = 60
    
    # Vehicle spacing (increased for better collision avoidance)
    STOPPING_GAP = 20
    MOVING_GAP = 25
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
# CORE CLASSES
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
        
        # Timing
        self.spawn_time = SimulationManager.time_elapsed
        self.cross_time = None
        
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
        """Load vehicle image or create placeholder."""
        image_path = f"images/{self.direction}/{self.vehicle_class}.png"
        
        if os.path.exists(image_path):
            self.original_image = pygame.image.load(image_path)
            self.current_image = pygame.image.load(image_path)
        else:
            # Create colored placeholder
            width, height = (40, 20) if self.vehicle_class == 'car' else (30, 15)
            self.original_image = pygame.Surface((width, height))
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
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
        """Update vehicle position based on traffic rules and signal state."""
        if self.direction == 'right':
            self._move_right()
        elif self.direction == 'down':
            self._move_down()
        elif self.direction == 'left':
            self._move_left()
        elif self.direction == 'up':
            self._move_up()
    
    def _move_right(self):
        """Handle movement for vehicles going right."""
        # Check if vehicle has crossed the stop line
        if (self.crossed == 0 and 
            self.x + self.current_image.get_rect().width > IntersectionLayout.STOP_LINES[self.direction]):
            self._mark_crossed()
        
        if self.will_turn:
            self._handle_right_turn()
        else:
            self._move_straight_right()
    
    def _move_down(self):
        """Handle movement for vehicles going down."""
        if (self.crossed == 0 and 
            self.y + self.current_image.get_rect().height > IntersectionLayout.STOP_LINES[self.direction]):
            self._mark_crossed()
        
        if self.will_turn:
            self._handle_down_turn()
        else:
            self._move_straight_down()
    
    def _move_left(self):
        """Handle movement for vehicles going left."""
        if self.crossed == 0 and self.x < IntersectionLayout.STOP_LINES[self.direction]:
            self._mark_crossed()
        
        if self.will_turn:
            self._handle_left_turn()
        else:
            self._move_straight_left()
    
    def _move_up(self):
        """Handle movement for vehicles going up."""
        if self.crossed == 0 and self.y < IntersectionLayout.STOP_LINES[self.direction]:
            self._mark_crossed()
        
        if self.will_turn:
            self._handle_up_turn()
        else:
            self._move_straight_up()
    
    def _mark_crossed(self):
        """Mark vehicle as having crossed the intersection."""
        self.crossed = 1
        self.cross_time = SimulationManager.time_elapsed
        SimulationManager.vehicles[self.direction]['crossed'] += 1
        SimulationManager.waiting_times[self.direction_number].append(
            self.cross_time - self.spawn_time
        )
    
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
        """Check if there's enough space to move without hitting vehicle ahead."""
        vehicles_in_lane = SimulationManager.vehicles[self.direction][self.lane]
        
        if self.index == 0:  # First vehicle in lane
            return True
            
        ahead_vehicle = vehicles_in_lane[self.index - 1]
        if ahead_vehicle.crossed == 1 and ahead_vehicle.turned == 1:
            return True  # Vehicle ahead has cleared the lane
        
        safe_distance = Config.MOVING_GAP + 5  # Extra safety margin
        
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
# SIMULATION MANAGEMENT
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
    
    # Analytics
    waiting_times = {0: [], 1: [], 2: [], 3: []}
    queue_history = {0: [], 1: [], 2: [], 3: []}
    time_stamps = []
    last_served = {0: 0, 1: 0, 2: 0, 3: 0}
    results = {
        "served_counts": {0: 0, 1: 0, 2: 0, 3: 0},
        "fairness_used": 0,
        "adaptive_used": 0
    }
    
    # Pygame components
    simulation = pygame.sprite.Group()
    
    @classmethod
    def initialize(cls):
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
        
        # Start the signal control thread
        signal_thread = threading.Thread(target=cls._signal_control_loop, daemon=True)
        signal_thread.start()
    
    @classmethod
    def calculate_required_time(cls, direction_idx: int) -> int:
        """Calculate required green time based on vehicle count and types."""
        direction = Vehicle.DIRECTION_NUMBERS[direction_idx]
        vehicle_counts = {'car': 0, 'bus': 0, 'truck': 0, 'rickshaw': 0, 'bike': 0}
        
        # Count vehicles by type
        for lane in range(3):
            for vehicle in cls.vehicles[direction][lane]:
                if vehicle.crossed == 0:
                    vehicle_counts[vehicle.vehicle_class] += 1
        
        total_vehicles = sum(vehicle_counts.values())
        if total_vehicles == 0:
            return 0
        
        # Calculate required time based on vehicle processing times
        total_time = sum(count * Config.VEHICLE_TIMES[vtype] 
                        for vtype, count in vehicle_counts.items())
        
        green_time = math.ceil(total_time / Config.NUM_LANES)
        
        # Apply minimum and maximum constraints
        return max(Config.DEFAULT_MINIMUM, 
                  min(green_time, Config.DEFAULT_MAXIMUM))
    
    @classmethod
    def _signal_control_loop(cls):
        """Main signal control loop with adaptive timing and fairness."""
        while True:
            # Green phase
            while cls.signals[cls.current_green].green > 0:
                cls._update_signal_timers()
                time.sleep(1)
            
            # Yellow phase
            cls.current_yellow = 1
            cls._reset_vehicle_stops()
            
            while cls.signals[cls.current_green].yellow > 0:
                cls._update_signal_timers()
                time.sleep(1)
            
            # End of cycle - choose next signal adaptively
            cls.current_yellow = 0
            cls._reset_current_signal()
            cls._select_next_signal_adaptive()
            
            # Update current signal
            cls.current_green = cls.next_green
            cls.next_green = (cls.current_green + 1) % Config.NUM_SIGNALS
            cls._update_red_times()
    
    @classmethod
    def _update_signal_timers(cls):
        """Update signal timers and record analytics."""
        cls.time_stamps.append(cls.time_elapsed)
        
        # Record queue lengths
        for direction_idx in range(Config.NUM_SIGNALS):
            direction = Vehicle.DIRECTION_NUMBERS[direction_idx]
            queue_count = sum(len([v for v in cls.vehicles[direction][lane] if v.crossed == 0])
                            for lane in range(3))
            cls.queue_history[direction_idx].append(queue_count)
        
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
    def _select_next_signal_adaptive(cls):
        """Select next signal using adaptive algorithm with fairness."""
        cls.cycle_counter += 1
        
        # Check fairness constraints
        forced_direction = cls._check_fairness_constraints()
        
        if forced_direction is not None:
            cls._apply_fairness_selection(forced_direction)
        else:
            cls._apply_adaptive_selection()
        
        cls.last_served[cls.next_green] = cls.cycle_counter
        cls.results['served_counts'][cls.next_green] = cls.vehicles[
            Vehicle.DIRECTION_NUMBERS[cls.next_green]]['crossed']
        
        cls._log_selection_decision()
    
    @classmethod
    def _check_fairness_constraints(cls) -> Optional[int]:
        """Check if any direction needs fairness enforcement."""
        for direction in range(Config.NUM_SIGNALS):
            cycles_waiting = cls.cycle_counter - cls.last_served[direction]
            if (cycles_waiting >= Config.FAIRNESS_LIMITS[direction] and 
                cls.calculate_required_time(direction) > 0):
                return direction
        return None
    
    @classmethod
    def _apply_fairness_selection(cls, forced_direction: int):
        """Apply fairness-based signal selection."""
        cls.next_green = forced_direction
        green_time = cls.calculate_required_time(forced_direction)
        green_time = max(green_time, Config.DEFAULT_MINIMUM)
        green_time = min(green_time, Config.DEFAULT_MINIMUM + 5)  # Fairness clamp
        
        cls.signals[cls.next_green].green = green_time
        cls.signals[cls.next_green].yellow = Config.DEFAULT_YELLOW
        cls.signals[cls.next_green].red = Config.DEFAULT_RED
        cls.results['fairness_used'] += 1
    
    @classmethod
    def _apply_adaptive_selection(cls):
        """Apply adaptive signal selection based on demand."""
        best_direction = -1
        best_time = -1
        
        for direction in range(Config.NUM_SIGNALS):
            required_time = cls.calculate_required_time(direction)
            if required_time > 0 and required_time > best_time:
                best_time = required_time
                best_direction = direction
        
        if best_direction == -1:
            # No demand, use round-robin
            cls.next_green = (cls.current_green + 1) % Config.NUM_SIGNALS
            green_time = Config.DEFAULT_GREEN
        else:
            cls.next_green = best_direction
            green_time = best_time
        
        cls.signals[cls.next_green].green = green_time
        cls.signals[cls.next_green].yellow = Config.DEFAULT_YELLOW
        cls.signals[cls.next_green].red = Config.DEFAULT_RED
        cls.results['adaptive_used'] += 1
    
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
        
        demands = [cls.calculate_required_time(d) for d in range(Config.NUM_SIGNALS)]
        
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
            
            time.sleep(1.0)  # Slower generation rate to prevent overcrowding


# ============================================================================
# ANALYTICS AND REPORTING
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
        
        # Calculate overall statistics
        all_waiting_times = []
        for direction_times in SimulationManager.waiting_times.values():
            all_waiting_times.extend(direction_times)
        
        overall_avg_wait = (sum(all_waiting_times) / len(all_waiting_times)) if all_waiting_times else 0.0
        throughput = (total_vehicles / SimulationManager.time_elapsed) * 60.0 if SimulationManager.time_elapsed > 0 else 0.0
        
        print(f'Overall average wait time: {overall_avg_wait:.2f}s')
        print(f'Throughput: {throughput:.2f} vehicles/minute')
        
        print(f'\nAdaptive decisions: {SimulationManager.results["adaptive_used"]}')
        print(f'Fairness enforcements: {SimulationManager.results["fairness_used"]}')
        
        # Generate plots
        AnalyticsManager._generate_plots()
    
    @staticmethod
    def _generate_plots():
        """Generate all analytical plots in a single side-by-side dashboard."""
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
# VISUALIZATION AND UI
# ============================================================================

class VisualizationManager:
    """Manages the Pygame-based visualization."""
    
    # Color constants
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    YELLOW = (255, 255, 0)
    GREEN = (0, 200, 0)
    
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
        """Load background and signal images."""
        try:
            self.background = pygame.image.load('images/mod_int.png')
        except Exception:
            self.background = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
            self.background.fill((50, 50, 50))
        
        try:
            self.red_signal = pygame.image.load('images/signals/red.png')
            self.yellow_signal = pygame.image.load('images/signals/yellow.png')
            self.green_signal = pygame.image.load('images/signals/green.png')
        except Exception:
            self.red_signal = self.yellow_signal = self.green_signal = None
    
    def _load_fonts(self):
        """Load fonts for text rendering."""
        self.font = pygame.font.Font(None, 30)
        self.large_font = pygame.font.Font(None, 36)
    
    def render_frame(self):
        """Render a single frame of the simulation."""
        # Clear screen
        self.screen.blit(self.background, (0, 0))
        
        # Render traffic signals
        self._render_signals()
        
        # Render vehicles
        self._render_vehicles()
        
        # Render UI elements
        self._render_ui()
        
        # Update display
        pygame.display.update()
        self.clock.tick(Config.FPS)
    
    def _render_signals(self):
        """Render traffic signal displays."""
        for i in range(Config.NUM_SIGNALS):
            signal = SimulationManager.signals[i]
            coord = IntersectionLayout.SIGNAL_COORDS[i]
            
            if i == SimulationManager.current_green:
                if SimulationManager.current_yellow == 1:
                    # Yellow signal
                    self._draw_signal_light(coord, self.yellow_signal, self.YELLOW)
                    signal.signal_text = str(signal.yellow) if signal.yellow > 0 else "STOP"
                else:
                    # Green signal
                    self._draw_signal_light(coord, self.green_signal, self.GREEN)
                    signal.signal_text = str(signal.green) if signal.green > 0 else "SLOW"
            else:
                # Red signal
                self._draw_signal_light(coord, self.red_signal, self.RED)
                if signal.red <= 10:
                    signal.signal_text = str(signal.red) if signal.red > 0 else "WAIT"
                else:
                    signal.signal_text = "---"
            
            # Render signal timer text
            timer_coord = IntersectionLayout.SIGNAL_TIMER_COORDS[i]
            timer_text = self.font.render(str(signal.signal_text), True, self.WHITE, self.BLACK)
            self.screen.blit(timer_text, timer_coord)
    
    def _draw_signal_light(self, coord: Tuple[int, int], image: Optional[pygame.Surface], color: Tuple[int, int, int]):
        """Draw a traffic signal light."""
        if image:
            self.screen.blit(image, coord)
        else:
            pygame.draw.circle(self.screen, color, coord, 20)
    
    def _render_vehicles(self):
        """Render all vehicles in the simulation."""
        for vehicle in SimulationManager.simulation:
            try:
                self.screen.blit(vehicle.current_image, (vehicle.x, vehicle.y))
                vehicle.move()
            except Exception:
                # Handle any rendering errors gracefully
                pass
    
    def _render_ui(self):
        """Render user interface elements."""
        # Vehicle count displays
        for i in range(Config.NUM_SIGNALS):
            direction_name = Vehicle.DIRECTION_NUMBERS[i]
            crossed_count = SimulationManager.vehicles[direction_name]['crossed']
            
            count_coord = IntersectionLayout.VEHICLE_COUNT_COORDS[i]
            count_text = self.font.render(str(crossed_count), True, self.BLACK, self.WHITE)
            self.screen.blit(count_text, count_coord)
        
        # Elapsed time display
        time_text = self.font.render(f"Time Elapsed: {SimulationManager.time_elapsed}s", 
                                   True, self.BLACK, self.WHITE)
        self.screen.blit(time_text, (1100, 50))
        
        # Simulation info
        cycle_text = self.font.render(f"Cycle: {SimulationManager.cycle_counter}", 
                                    True, self.BLACK, self.WHITE)
        self.screen.blit(cycle_text, (1100, 80))
        
        # Current phase indicator
        current_direction = Vehicle.DIRECTION_NUMBERS[SimulationManager.current_green].capitalize()
        phase_color = "YELLOW" if SimulationManager.current_yellow else "GREEN"
        phase_text = self.font.render(f"Active: {current_direction} ({phase_color})", 
                                    True, self.BLACK, self.WHITE)
        self.screen.blit(phase_text, (1100, 110))


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class TrafficSimulationApp:
    """Main application class that orchestrates the entire simulation."""
    
    def __init__(self):
        self.visualization = VisualizationManager()
        self.running = True
    
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
        """Main simulation loop with Pygame event handling."""
        await self.start_simulation_async()
        
        while self.running:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self._print_current_status()
                    elif event.key == pygame.K_r:
                        self._print_analytics_summary()
            
            # Render frame
            self.visualization.render_frame()
            
            # Async yield for smooth animation
            await asyncio.sleep(1.0 / Config.FPS)
    
    async def start_simulation_async(self):
        """Async wrapper for simulation initialization."""
        self.start_simulation()
    
    def _print_current_status(self):
        """Print current simulation status (triggered by spacebar)."""
        print(f"\n--- Current Status (t={SimulationManager.time_elapsed}s) ---")
        print(f"Cycle: {SimulationManager.cycle_counter}")
        print(f"Current Green: {Vehicle.DIRECTION_NUMBERS[SimulationManager.current_green].capitalize()}")
        
        for i in range(Config.NUM_SIGNALS):
            direction = Vehicle.DIRECTION_NUMBERS[i]
            queue_size = sum(len([v for v in SimulationManager.vehicles[direction][lane] if v.crossed == 0])
                           for lane in range(3))
            served = SimulationManager.vehicles[direction]['crossed']
            print(f"  {direction.capitalize()}: Queue={queue_size}, Served={served}")
    
    def _print_analytics_summary(self):
        """Print analytics summary (triggered by 'r' key)."""
        print(f"\n--- Analytics Summary (t={SimulationManager.time_elapsed}s) ---")
        
        total_served = sum(SimulationManager.vehicles[Vehicle.DIRECTION_NUMBERS[i]]['crossed'] 
                          for i in range(Config.NUM_SIGNALS))
        
        print(f"Total vehicles served: {total_served}")
        print(f"Adaptive decisions: {SimulationManager.results['adaptive_used']}")
        print(f"Fairness enforcements: {SimulationManager.results['fairness_used']}")
        
        if SimulationManager.time_elapsed > 0:
            throughput = (total_served / SimulationManager.time_elapsed) * 60
            print(f"Current throughput: {throughput:.2f} vehicles/minute")


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


if __name__ == "__main__":
    # Handle different execution environments
    if platform.system() == "Emscripten":
        # For web deployment
        asyncio.ensure_future(main())
    else:
        # For desktop execution
        asyncio.run(main())


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


# Print configuration info when module is imported
if __name__ == "__main__":
    print_simulation_info()