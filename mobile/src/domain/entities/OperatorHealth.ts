export type OperatorHealth = {
  status: string;
  service: string;
  serviceHealth: string;
  loginEnabled: boolean;
  cloudflareAccessEnabled: boolean;
  apiTokenEnabled: boolean;
};
