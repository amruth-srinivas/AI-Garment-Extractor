import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import JobPage from "./pages/Job";
import Results from "./pages/Results";
import Workspace from "./pages/Workspace";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="new" element={<Workspace />} />
        <Route path="jobs/:jobId" element={<JobPage />} />
        <Route path="results/:resultId" element={<Results />} />
      </Route>
    </Routes>
  );
}
