import React from 'react';
import RoomBriefing from './RoomBriefing.jsx';

export default function FootballView({ room }) {
  return (
    <RoomBriefing room={room}
      legend={[
        ['#0369a1', 'Striker (you) — continuous x, y, Vx, Vy'],
        ['#b91c1c', 'Defenders — chase the ball carrier (tackle −50)'],
        ['#ca8a04', 'Keeper — patrols the goal mouth side to side'],
        ['#34d399', 'Shooting zone — kicks only count from here'],
        ['#f8fafc', 'Ball — physically flies with power + curve'],
        ['#0a3320', 'Pitch — out of bounds ends the move'],
      ]}
      tips={[
        'The state is continuous, so no Q-table can exist — a neural network (DQN) approximates Q(s, a).',
        'The kick is a real physical event: the ball leaves the player and the keeper keeps patrolling during the flight.',
        'Timing beats power: the network learns to shoot when the keeper drifts away, or to curve around him.',
        'Free-kick mode adds a third dimension — chip the ball over the wall or bend it around.',
      ]} />
  );
}
