import React, { useEffect, useState } from 'react';
import { Route, Routes } from 'react-router-dom';
import Layout from './components/Layout.jsx';
import Home from './pages/Home.jsx';
import RoomsPage from './pages/RoomsPage.jsx';
import RoomPage from './pages/RoomPage.jsx';
import AlgorithmsPage from './pages/AlgorithmsPage.jsx';
import ResultsPage from './pages/ResultsPage.jsx';
import { api } from './api.js';

export default function App() {
  const [anyTraining, setAnyTraining] = useState(false);

  // HUD LED lights amber while any room is training
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const { rooms } = await api.rooms();
        if (alive) setAnyTraining(rooms.some((r) => r.training));
      } catch { /* backend offline — LED stays green */ }
    };
    tick();
    const t = setInterval(tick, 4000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  return (
    <Layout training={anyTraining}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/rooms" element={<RoomsPage />} />
        <Route path="/room/:id" element={<RoomPage />} />
        <Route path="/algorithms" element={<AlgorithmsPage />} />
        <Route path="/results" element={<ResultsPage />} />
      </Routes>
    </Layout>
  );
}
