import { StatusBar } from "expo-status-bar";

import { TradingControlScreen } from "./src/interface/screens/TradingControlScreen";

export default function App() {
  return (
    <>
      <TradingControlScreen />
      <StatusBar style="light" />
    </>
  );
}
