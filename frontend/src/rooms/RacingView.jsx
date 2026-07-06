import React from 'react';
import RoomBriefing from './RoomBriefing.jsx';

export default function RacingView({ room }) {
  return (
    <RoomBriefing room={room}
      legend={[
        ['#38bdf8', 'Booster pad — one-time +20, unlocks the finish'],
        ['#0c0d12', 'Oil slick — slippery; a slip into a wall = crash'],
        ['#8b5a2b', 'Mud — slows the car (−5)'],
        ['#d97706', 'Crash barrier — race over (−200)'],
        ['#e5e7eb', 'Finish line — locked until enough boosters'],
        ['#1b2030', 'City block — track boundary'],
      ]}
      tips={[
        'Q-Learning is off-policy: it learns the greedy optimal line even while the behaviour policy still explores.',
        'The finish stays locked until the booster quota is met — the racing line must detour through pads.',
        'Oil creates a classic risk/return dilemma: the short line crosses slicks, the safe line is longer.',
        'Watch early replays crash constantly, then the racing line sharpen as ε decays.',
      ]} />
  );
}
