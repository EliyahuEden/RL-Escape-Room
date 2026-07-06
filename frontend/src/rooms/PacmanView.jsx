import React from 'react';
import RoomBriefing from './RoomBriefing.jsx';

export default function PacmanView({ room }) {
  return (
    <RoomBriefing room={room}
      legend={[
        ['#ffd23f', 'Coin — collect ALL of them (+10)'],
        ['#2649ff', 'Maze wall — blocks movement'],
        ['#7dd3fc', 'Ice — may deflect the move sideways'],
        ['#34d399', 'Exit door — unlocks after the last coin'],
        ['#ff4d5a', 'Ghost guard — patrols or chases (−50)'],
        ['#0a0f22', 'Corridor — every step costs −1'],
      ]}
      tips={[
        'The model is fully known, so Value/Policy Iteration solve the maze offline — no trial and error.',
        'The state includes the coin bitmask: the same cell has a different optimal action depending on which coins remain.',
        'Watch the policy view flip once all coins are collected — the whole value landscape re-points to the door.',
        'Ice tiles make transitions stochastic; DP handles them through expected values, not luck.',
      ]} />
  );
}
