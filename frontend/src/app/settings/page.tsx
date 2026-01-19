"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { PageHeader, StatusBadge } from "@/components/page-header";
import { MotionWrapper, StaggerContainer, StaggerItem } from "@/components/motion-wrapper";
import {
  Settings,
  Palette,
  Bell,
  Monitor,
  Moon,
  Sun,
  Volume2,
  VolumeX,
  Eye,
  EyeOff,
  RefreshCw,
  Save,
  Zap,
  Shield,
  Clock,
  Server,
} from "lucide-react";
import { toast } from "sonner";
import {
  STORAGE_KEYS,
  getStorageItem,
  setStorageItem,
  removeStorageItem,
  getStorageJSON,
  setStorageJSON,
  migrateStorageKeys,
} from "@/lib/storage";

interface SettingToggleProps {
  label: string;
  description: string;
  enabled: boolean;
  onChange: (enabled: boolean) => void;
  icon: React.ReactNode;
}

interface SettingsData {
  soundEnabled?: boolean;
  desktopNotifications?: boolean;
  violationAlerts?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: string;
  showAnimations?: boolean;
}

function SettingToggle({ label, description, enabled, onChange, icon }: SettingToggleProps) {
  return (
    <motion.div
      whileHover={{ x: 4 }}
      className="flex items-center justify-between p-4 rounded-xl border border-border/50 hover:border-border transition-all duration-200 bg-card/50"
    >
      <div className="flex items-center gap-4">
        <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-primary/10">
          {icon}
        </div>
        <div>
          <p className="font-medium text-sm">{label}</p>
          <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
        </div>
      </div>
      <motion.button
        whileTap={{ scale: 0.95 }}
        onClick={() => onChange(!enabled)}
        className={`relative w-12 h-7 rounded-full transition-colors duration-200 ${enabled ? "bg-primary" : "bg-muted"
          }`}
      >
        <motion.div
          animate={{ x: enabled ? 22 : 4 }}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
          className="absolute top-1 w-5 h-5 rounded-full bg-white shadow-md"
        />
      </motion.button>
    </motion.div>
  );
}

interface SettingSelectProps {
  label: string;
  description: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
  icon: React.ReactNode;
}

function SettingSelect({ label, description, value, options, onChange, icon }: SettingSelectProps) {
  return (
    <motion.div
      whileHover={{ x: 4 }}
      className="flex items-center justify-between p-4 rounded-xl border border-border/50 hover:border-border transition-all duration-200 bg-card/50"
    >
      <div className="flex items-center gap-4">
        <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-primary/10">
          {icon}
        </div>
        <div>
          <p className="font-medium text-sm">{label}</p>
          <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
        </div>
      </div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="px-3 py-1.5 rounded-lg border border-border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </motion.div>
  );
}

export default function SettingsPage() {
  // Lazy state initialization with safe defaults
  const [theme, setTheme] = useState<"light" | "dark" | "system">(() => {
    if (typeof window !== "undefined") {
      migrateStorageKeys();
      const savedTheme = getStorageItem(STORAGE_KEYS.THEME);
      if (savedTheme === "light" || savedTheme === "dark") {
        return savedTheme;
      }
    }
    return "system";
  });

  // Notification settings with lazy initialization
  const [soundEnabled, setSoundEnabled] = useState(() => {
    if (typeof window !== "undefined") {
      const settings = getStorageJSON<SettingsData>(STORAGE_KEYS.SETTINGS, {});
      return settings.soundEnabled ?? true;
    }
    return true;
  });

  const [desktopNotifications, setDesktopNotifications] = useState(() => {
    if (typeof window !== "undefined") {
      const settings = getStorageJSON<SettingsData>(STORAGE_KEYS.SETTINGS, {});
      return settings.desktopNotifications ?? true;
    }
    return true;
  });

  const [violationAlerts, setViolationAlerts] = useState(() => {
    if (typeof window !== "undefined") {
      const settings = getStorageJSON<SettingsData>(STORAGE_KEYS.SETTINGS, {});
      return settings.violationAlerts ?? true;
    }
    return true;
  });

  // Display settings with lazy initialization
  const [autoRefresh, setAutoRefresh] = useState(() => {
    if (typeof window !== "undefined") {
      const settings = getStorageJSON<SettingsData>(STORAGE_KEYS.SETTINGS, {});
      return settings.autoRefresh ?? true;
    }
    return true;
  });

  const [refreshInterval, setRefreshInterval] = useState(() => {
    if (typeof window !== "undefined") {
      const settings = getStorageJSON<SettingsData>(STORAGE_KEYS.SETTINGS, {});
      return settings.refreshInterval ?? "30";
    }
    return "30";
  });

  const [showAnimations, setShowAnimations] = useState(() => {
    if (typeof window !== "undefined") {
      const settings = getStorageJSON<SettingsData>(STORAGE_KEYS.SETTINGS, {});
      return settings.showAnimations ?? true;
    }
    return true;
  });

  const handleThemeChange = (newTheme: "light" | "dark" | "system") => {
    setTheme(newTheme);

    if (newTheme === "system") {
      const systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      document.documentElement.classList.remove("light", "dark");
      document.documentElement.classList.add(systemDark ? "dark" : "light");
      removeStorageItem(STORAGE_KEYS.THEME);
    } else {
      document.documentElement.classList.remove("light", "dark");
      document.documentElement.classList.add(newTheme);
      setStorageItem(STORAGE_KEYS.THEME, newTheme);
    }

    toast.success("Theme Updated", {
      description: `Switched to ${newTheme} mode`,
    });
  };

  const handleSaveSettings = () => {
    const settings = {
      soundEnabled,
      desktopNotifications,
      violationAlerts,
      autoRefresh,
      refreshInterval,
      showAnimations,
    };

    const success = setStorageJSON(STORAGE_KEYS.SETTINGS, settings);

    if (success) {
      toast.success("Settings Saved", {
        description: "Your preferences have been saved successfully",
        icon: <Save className="w-4 h-4" />,
      });
    } else {
      toast.error("Failed to Save Settings", {
        description: "Could not save preferences. Storage may be disabled.",
      });
    }
  };

  const handleResetSettings = () => {
    setSoundEnabled(true);
    setDesktopNotifications(true);
    setViolationAlerts(true);
    setAutoRefresh(true);
    setRefreshInterval("30");
    setShowAnimations(true);
    removeStorageItem(STORAGE_KEYS.SETTINGS);

    toast.info("Settings Reset", {
      description: "All settings have been restored to defaults",
    });
  };

  return (
    <MotionWrapper className="space-y-8">
      {/* Header */}
      <PageHeader
        title="Settings"
        description="Customize your experience"
        action={
          <div className="flex items-center gap-2">
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button variant="outline" size="sm" onClick={handleResetSettings}>
                <RefreshCw className="w-4 h-4 mr-2" />
                Reset
              </Button>
            </motion.div>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button size="sm" onClick={handleSaveSettings}>
                <Save className="w-4 h-4 mr-2" />
                Save Changes
              </Button>
            </motion.div>
          </div>
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Appearance Section */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card variant="glass" hover>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <motion.div
                  whileHover={{ scale: 1.1, rotate: 10 }}
                  className="flex items-center justify-center w-10 h-10 rounded-xl bg-primary/10"
                >
                  <Palette className="w-5 h-5 text-primary" />
                </motion.div>
                <div>
                  <span>Appearance</span>
                  <CardDescription className="mt-0.5">
                    Customize how SentinelVision looks
                  </CardDescription>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Theme Selection */}
              <div className="space-y-3">
                <p className="text-sm font-medium text-muted-foreground">Theme</p>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { value: "light", label: "Light", icon: Sun },
                    { value: "dark", label: "Dark", icon: Moon },
                    { value: "system", label: "System", icon: Monitor },
                  ].map(({ value, label, icon: Icon }) => (
                    <motion.button
                      key={value}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => handleThemeChange(value as "light" | "dark" | "system")}
                      className={`flex flex-col items-center gap-2 p-4 rounded-xl border transition-all duration-200 ${theme === value
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border hover:border-primary/50"
                        }`}
                    >
                      <Icon className="w-5 h-5" />
                      <span className="text-xs font-medium">{label}</span>
                    </motion.button>
                  ))}
                </div>
              </div>

              <SettingToggle
                label="Show Animations"
                description="Enable smooth transitions and effects"
                enabled={showAnimations}
                onChange={setShowAnimations}
                icon={<Zap className="w-5 h-5 text-primary" />}
              />
            </CardContent>
          </Card>
        </motion.div>

        {/* Notifications Section */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card variant="glass" hover>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <motion.div
                  whileHover={{ scale: 1.1, rotate: -10 }}
                  className="flex items-center justify-center w-10 h-10 rounded-xl bg-warning/10"
                >
                  <Bell className="w-5 h-5 text-warning" />
                </motion.div>
                <div>
                  <span>Notifications</span>
                  <CardDescription className="mt-0.5">
                    Configure alert preferences
                  </CardDescription>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <SettingToggle
                label="Violation Alerts"
                description="Get notified when violations are detected"
                enabled={violationAlerts}
                onChange={setViolationAlerts}
                icon={<Shield className="w-5 h-5 text-danger" />}
              />
              <SettingToggle
                label="Sound Notifications"
                description="Play sound for important alerts"
                enabled={soundEnabled}
                onChange={setSoundEnabled}
                icon={soundEnabled ? <Volume2 className="w-5 h-5 text-info" /> : <VolumeX className="w-5 h-5 text-muted-foreground" />}
              />
              <SettingToggle
                label="Desktop Notifications"
                description="Show browser notifications"
                enabled={desktopNotifications}
                onChange={setDesktopNotifications}
                icon={desktopNotifications ? <Eye className="w-5 h-5 text-success" /> : <EyeOff className="w-5 h-5 text-muted-foreground" />}
              />
            </CardContent>
          </Card>
        </motion.div>

        {/* Data & Performance Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <Card variant="glass" hover>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <motion.div
                  whileHover={{ scale: 1.1 }}
                  className="flex items-center justify-center w-10 h-10 rounded-xl bg-success/10"
                >
                  <Server className="w-5 h-5 text-success" />
                </motion.div>
                <div>
                  <span>Data & Performance</span>
                  <CardDescription className="mt-0.5">
                    Manage data refresh and performance
                  </CardDescription>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <SettingToggle
                label="Auto-Refresh Data"
                description="Automatically refresh dashboard data"
                enabled={autoRefresh}
                onChange={setAutoRefresh}
                icon={<RefreshCw className="w-5 h-5 text-primary" />}
              />
              <SettingSelect
                label="Refresh Interval"
                description="How often to refresh data"
                value={refreshInterval}
                onChange={setRefreshInterval}
                options={[
                  { value: "10", label: "10 seconds" },
                  { value: "30", label: "30 seconds" },
                  { value: "60", label: "1 minute" },
                  { value: "300", label: "5 minutes" },
                ]}
                icon={<Clock className="w-5 h-5 text-info" />}
              />
            </CardContent>
          </Card>
        </motion.div>

        {/* About Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <Card variant="glass" hover>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <motion.div
                  whileHover={{ scale: 1.1, rotate: 360 }}
                  transition={{ duration: 0.5 }}
                  className="flex items-center justify-center w-10 h-10 rounded-xl bg-info/10"
                >
                  <Settings className="w-5 h-5 text-info" />
                </motion.div>
                <div>
                  <span>About</span>
                  <CardDescription className="mt-0.5">
                    System information
                  </CardDescription>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between p-4 rounded-xl bg-muted/30">
                <div>
                  <p className="font-medium text-sm">Lab Safety</p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    AI-powered safety compliance monitoring
                  </p>
                </div>
                <Badge variant="info">v1.0.0</Badge>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-muted/30 text-center">
                  <p className="text-2xl font-bold text-primary">YOLOv11</p>
                  <p className="text-xs text-muted-foreground">PPE Detection</p>
                </div>
                <div className="p-3 rounded-xl bg-muted/30 text-center">
                  <p className="text-2xl font-bold text-success">SAM3</p>
                  <p className="text-xs text-muted-foreground">Segmentation</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </MotionWrapper>
  );
}
