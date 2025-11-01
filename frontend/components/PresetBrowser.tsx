import React, { useState, useEffect } from 'react';
import { notify } from './UI/Notification';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Visibility as PreviewIcon,
  Save as SaveIcon,
  Share as ShareIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
  Info as InfoIcon
} from '@mui/icons-material';

// ============================================================================
// Types
// ============================================================================

interface PresetNode {
  id: string;
  type: string;
  name: string;
  position: { x: number; y: number };
  config: Record<string, any>;
}

interface PresetConnection {
  from: string;
  to: string;
  description?: string;
}

interface PresetControl {
  name: string;
  label: string;
  node: string;
  field: string;
  type: 'slider' | 'select' | 'discrete' | 'preset';
  min?: number;
  max?: number;
  step?: number;
  options?: any;
  default?: any;
}

interface Preset {
  id: string;
  name: string;
  version: string;
  description: string;
  category: string;
  tags: string[];
  author: string;
  created: string;
  nodes: PresetNode[];
  connections: PresetConnection[];
  controls?: {
    description: string;
    parameters: PresetControl[];
  };
  requirements?: {
    minGPUMemory?: string;
    recommendedGPUMemory?: string;
    [key: string]: any;
  };
  documentation?: {
    overview: string;
    expectedBehavior: string;
    parameters?: Record<string, string>;
    successMetrics?: string[];
  };
}

// ============================================================================
// PresetBrowser Component
// ============================================================================

export const PresetBrowser: React.FC = () => {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<Preset | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [loadingPreset, setLoadingPreset] = useState(false);

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<string>('All');
  const [sortBy, setSortBy] = useState<string>('name');

  // Load presets
  useEffect(() => {
    loadPresets();
  }, []);

  const loadPresets = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/presets');
      if (!response.ok) {
        throw new Error('Failed to load presets');
      }
      const data = await response.json();
      setPresets(data.presets || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // Filter and sort presets
  const filteredPresets = presets
    .filter(preset => {
      const matchesSearch = 
        preset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        preset.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        preset.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
      
      const matchesCategory = 
        categoryFilter === 'All' || preset.category === categoryFilter;
      
      return matchesSearch && matchesCategory;
    })
    .sort((a, b) => {
      if (sortBy === 'name') return a.name.localeCompare(b.name);
      if (sortBy === 'category') return a.category.localeCompare(b.category);
      if (sortBy === 'created') return new Date(b.created).getTime() - new Date(a.created).getTime();
      return 0;
    });

  // Get unique categories
  const categories = ['All', ...Array.from(new Set(presets.map(p => p.category)))];

  // Load preset into system
  const handleLoadPreset = async (preset: Preset) => {
    try {
      setLoadingPreset(true);
      const response = await fetch(`/api/presets/${preset.id}/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(preset)
      });

      if (!response.ok) {
        throw new Error('Failed to load preset');
      }

      const result = await response.json();
      console.log('Preset loaded:', result);
      
      // Close preview and show success
      setPreviewOpen(false);
      notify.success(`Preset "${preset.name}" loaded successfully!`);
    } catch (err) {
      notify.error(`Error loading preset: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoadingPreset(false);
    }
  };

  // Preview preset
  const handlePreview = (preset: Preset) => {
    setSelectedPreset(preset);
    setPreviewOpen(true);
  };

  // Render preset card
  const renderPresetCard = (preset: Preset) => (
    <Card 
      key={preset.id}
      sx={{ 
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        transition: 'transform 0.2s, box-shadow 0.2s',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: 6
        }
      }}
    >
      <CardContent sx={{ flexGrow: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
          <Typography variant="h6" component="div" gutterBottom>
            {preset.name}
          </Typography>
          <Chip 
            label={preset.category} 
            size="small" 
            color={
              preset.category === 'AI/ML' ? 'primary' :
              preset.category === 'Creative' ? 'secondary' :
              'default'
            }
          />
        </Box>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 2, minHeight: 60 }}>
          {preset.description}
        </Typography>

        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
          {preset.tags.slice(0, 4).map(tag => (
            <Chip key={tag} label={tag} size="small" variant="outlined" />
          ))}
          {preset.tags.length > 4 && (
            <Chip label={`+${preset.tags.length - 4}`} size="small" variant="outlined" />
          )}
        </Box>

        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="caption" color="text.secondary">
            {preset.nodes.length} nodes • {preset.connections.length} connections
          </Typography>
        </Box>
      </CardContent>

      <Box sx={{ p: 2, pt: 0, display: 'flex', gap: 1 }}>
        <Button
          variant="contained"
          startIcon={<PlayIcon />}
          onClick={() => handleLoadPreset(preset)}
          fullWidth
        >
          Load
        </Button>
        <Button
          variant="outlined"
          startIcon={<PreviewIcon />}
          onClick={() => handlePreview(preset)}
        >
          Preview
        </Button>
      </Box>
    </Card>
  );

  // Render preview dialog
  const renderPreviewDialog = () => {
    if (!selectedPreset) return null;

    return (
      <Dialog 
        open={previewOpen} 
        onClose={() => setPreviewOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h5">{selectedPreset.name}</Typography>
            <Chip label={selectedPreset.category} color="primary" />
          </Box>
        </DialogTitle>

        <DialogContent dividers>
          {/* Overview */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom>Overview</Typography>
            <Typography variant="body1" paragraph>
              {selectedPreset.documentation?.overview || selectedPreset.description}
            </Typography>
          </Box>

          {/* Expected Behavior */}
          {selectedPreset.documentation?.expectedBehavior && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>Expected Behavior</Typography>
              <Typography variant="body1" paragraph>
                {selectedPreset.documentation.expectedBehavior}
              </Typography>
            </Box>
          )}

          {/* Node Graph */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom>Node Graph</Typography>
            <Box sx={{ 
              p: 2, 
              bgcolor: 'background.default', 
              borderRadius: 1,
              display: 'flex',
              flexDirection: 'column',
              gap: 1
            }}>
              <Typography variant="body2">
                <strong>Nodes:</strong> {selectedPreset.nodes.length}
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {selectedPreset.nodes.slice(0, 10).map(node => (
                  <Chip key={node.id} label={node.name} size="small" variant="outlined" />
                ))}
                {selectedPreset.nodes.length > 10 && (
                  <Chip label={`+${selectedPreset.nodes.length - 10} more`} size="small" />
                )}
              </Box>
              
              <Typography variant="body2" sx={{ mt: 1 }}>
                <strong>Connections:</strong> {selectedPreset.connections.length}
              </Typography>
            </Box>
          </Box>

          {/* Success Metrics */}
          {selectedPreset.documentation?.successMetrics && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>Success Metrics</Typography>
              <ul>
                {selectedPreset.documentation.successMetrics.map((metric, idx) => (
                  <li key={idx}>
                    <Typography variant="body2">{metric}</Typography>
                  </li>
                ))}
              </ul>
            </Box>
          )}

          {/* Requirements */}
          {selectedPreset.requirements && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>Requirements</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {selectedPreset.requirements.minGPUMemory && (
                  <Typography variant="body2">
                    <strong>Min GPU:</strong> {selectedPreset.requirements.minGPUMemory}
                  </Typography>
                )}
                {selectedPreset.requirements.recommendedGPUMemory && (
                  <Typography variant="body2">
                    <strong>Recommended GPU:</strong> {selectedPreset.requirements.recommendedGPUMemory}
                  </Typography>
                )}
              </Box>
            </Box>
          )}

          {/* Controls */}
          {selectedPreset.controls && selectedPreset.controls.parameters.length > 0 && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>Available Controls</Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                {selectedPreset.controls.description}
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {selectedPreset.controls.parameters.map(param => (
                  <Chip 
                    key={param.name} 
                    label={param.label} 
                    size="small" 
                    variant="outlined"
                    icon={<InfoIcon />}
                  />
                ))}
              </Box>
            </Box>
          )}
        </DialogContent>

        <DialogActions>
          <Button onClick={() => setPreviewOpen(false)}>Close</Button>
          <Button 
            variant="contained" 
            onClick={() => handleLoadPreset(selectedPreset)}
            startIcon={loadingPreset ? <CircularProgress size={20} /> : <PlayIcon />}
            disabled={loadingPreset}
          >
            {loadingPreset ? 'Loading...' : 'Load Preset'}
          </Button>
        </DialogActions>
      </Dialog>
    );
  };

  // Main render
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Application Presets
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Pre-configured node graphs for common applications. Click &quot;Load&quot; to instantly set up a complete system.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Filters */}
      <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
        <TextField
          placeholder="Search presets..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          InputProps={{
            startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />
          }}
          sx={{ minWidth: 300 }}
        />

        <FormControl sx={{ minWidth: 150 }}>
          <InputLabel>Category</InputLabel>
          <Select
            value={categoryFilter}
            label="Category"
            onChange={(e) => setCategoryFilter(e.target.value)}
          >
            {categories.map(cat => (
              <MenuItem key={cat} value={cat}>{cat}</MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl sx={{ minWidth: 150 }}>
          <InputLabel>Sort By</InputLabel>
          <Select
            value={sortBy}
            label="Sort By"
            onChange={(e) => setSortBy(e.target.value)}
          >
            <MenuItem value="name">Name</MenuItem>
            <MenuItem value="category">Category</MenuItem>
            <MenuItem value="created">Date</MenuItem>
          </Select>
        </FormControl>

        <Button
          variant="outlined"
          startIcon={<FilterIcon />}
          onClick={loadPresets}
        >
          Refresh
        </Button>
      </Box>

      {/* Preset Grid */}
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box sx={{ 
          display: 'grid', 
          gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: '1fr 1fr 1fr' },
          gap: 3 
        }}>
          {filteredPresets.map(preset => (
            <Box key={preset.id}>
              {renderPresetCard(preset)}
            </Box>
          ))}
        </Box>
      )}

      {!loading && filteredPresets.length === 0 && (
        <Box sx={{ textAlign: 'center', p: 4 }}>
          <Typography variant="h6" color="text.secondary">
            No presets found
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Try adjusting your search or filters
          </Typography>
        </Box>
      )}

      {/* Preview Dialog */}
      {renderPreviewDialog()}
    </Box>
  );
};

export default PresetBrowser;

