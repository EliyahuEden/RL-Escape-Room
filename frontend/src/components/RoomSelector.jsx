import React from 'react';
import RoomCard from './RoomCard.jsx';

/** The five level-select cards. */
export default function RoomSelector({ rooms }) {
  return (
    <div className="room-grid stagger">
      {rooms.map((room) => <RoomCard key={room.id} room={room} />)}
    </div>
  );
}
