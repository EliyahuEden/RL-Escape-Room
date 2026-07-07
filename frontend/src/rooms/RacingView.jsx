import React from 'react';
import RoomBriefing from './RoomBriefing.jsx';

export default function RacingView({ room }) {
  return (
    <RoomBriefing room={room}
      legend={[
        ['#ff4d5a', 'Your car — Q-Learning (off-policy)'],
        ['#7c5cd6', 'Rival car — SARSA (on-policy), same circuit & settings'],
        ['#38bdf8', 'Checkpoint gates — cross 1 then 2; every gate exists on BOTH routes'],
        ['#d21b2d', 'Crash barriers — one touch ends the race (−200)'],
        ['#c4a46e', 'Gravel trap — slows the car (−5)'],
        ['#0c0d12', 'Oil slick — slippery; a slip into a wall = crash'],
        ['#e5e7eb', 'Finish line — opens after all checkpoints; first car home wins'],
        ['#0d3d20', 'Grass infield — off the racing surface'],
      ]}
      tips={[
        'This track is CLIFF WALKING as a street race: the express lane hugs the barriers, the ring road detours around them.',
        'Q-Learning is off-policy — it backs up the greedy value, ignores its own exploration accidents, and learns the barrier-hugging express lane.',
        'SARSA is on-policy — every exploratory wobble into the barriers is priced into its Q-values, so it settles on the longer safe ring road.',
        'Both cars train with identical hyperparameters on the identical track. The race verdict is pure off-policy vs on-policy.',
        'Minimum ε stays high on purpose: with exploration annealed to zero the two algorithms would converge to the same line.',
      ]} />
  );
}
