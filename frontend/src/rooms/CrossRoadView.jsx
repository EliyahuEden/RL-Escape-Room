import React from 'react';
import RoomBriefing from './RoomBriefing.jsx';

export default function CrossRoadView({ room }) {
  return (
    <RoomBriefing room={room}
      legend={[
        ['#f8fafc', 'The chicken — cross left → right (+250)'],
        ['#ef4444', 'Traffic — collision ends the run (−140)'],
        ['#38bdf8', 'Sensor ring — the only thing the agent “sees”'],
        ['#34d399', 'Safe zone — the far sidewalk'],
        ['#101319', 'Road lanes — alternating directions, wrapping cars'],
        ['#252a36', 'Sidewalk — safe but time still costs reward'],
      ]}
      tips={[
        'The agent never sees the whole map — only the nearest cars inside sensor range (relative position, speed, closeness).',
        'Traffic wraps around and can be re-randomised every episode, so the policy must generalise, not memorise.',
        'The learned behaviour is readable in replays: dash, wait between lanes, let a car pass, dash again.',
        'Turn “new traffic every episode” off to see how much easier a memorisable world is.',
      ]} />
  );
}
