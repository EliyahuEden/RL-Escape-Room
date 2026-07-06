import React from 'react';
import RoomBriefing from './RoomBriefing.jsx';

export default function HeistView({ room }) {
  return (
    <RoomBriefing room={room}
      legend={[
        ['#38bdf8', 'Diamond — steal it from the vault (+30)'],
        ['#ff4d5a', 'Camera zone — detection −50 + triggers the alarm'],
        ['#fb923c', 'Laser trap — painful but survivable (−15)'],
        ['#fbbf24', 'Guard patrol — getting caught ends the heist (−50)'],
        ['#7dd3fc', 'Slippery marble — may deflect the move'],
        ['#34d399', 'Exit — only counts with the diamond in hand'],
      ]}
      tips={[
        'SARSA is on-policy: it learns the value of the ε-greedy policy it actually follows, so it naturally keeps a safety margin around cameras and guards.',
        'The alarm doubles guard speed for 5 turns — one careless camera step changes the whole endgame.',
        'The state includes the guard patrol phase: timing matters, not just position.',
        'Compare with Room 3: Q-Learning would hug danger more tightly than cautious SARSA.',
      ]} />
  );
}
