<template>
  <div class="contact-records">
    <el-page-header @back="goBack" :title="`联系记录 - ${expertName}`" />
    
    <el-card class="records-card">
      <template #header>
        <div class="card-header">
          <span>历史联系记录</span>
          <el-button type="primary" @click="showAddDialog">
            <el-icon><Plus /></el-icon>添加记录
          </el-button>
        </div>
      </template>
      
      <el-timeline v-loading="loading">
        <el-timeline-item
          v-for="record in records"
          :key="record.id"
          :timestamp="formatDate(record.contactTime)"
          :type="getTimelineType(record.contactMethod)"
        >
          <el-card class="record-card">
            <template #header>
              <div class="record-header">
                <span>{{ record.contactMethod || '其他' }}</span>
                <span class="operator">记录人: {{ record.operator || '未知' }}</span>
              </div>
            </template>
            <p>{{ record.content }}</p>
          </el-card>
        </el-timeline-item>
        
        <el-empty v-if="records.length === 0 && !loading" description="暂无联系记录" />
      </el-timeline>
    </el-card>
    
    <!-- 添加联系记录对话框 -->
    <el-dialog v-model="dialogVisible" title="添加联系记录" width="500px">
      <el-form :model="form" :rules="rules" ref="formRef" label-width="100px">
        <el-form-item label="联系时间" prop="contactTime">
          <el-date-picker
            v-model="form.contactTime"
            type="datetime"
            placeholder="选择联系时间"
            style="width: 100%"
          />
        </el-form-item>
        <el-form-item label="联系方式" prop="contactMethod">
          <el-select v-model="form.contactMethod" style="width: 100%">
            <el-option label="电话" value="电话" />
            <el-option label="邮件" value="邮件" />
            <el-option label="微信" value="微信" />
            <el-option label="面谈" value="面谈" />
            <el-option label="其他" value="其他" />
          </el-select>
        </el-form-item>
        <el-form-item label="联系内容" prop="content">
          <el-input
            v-model="form.content"
            type="textarea"
            :rows="4"
            placeholder="请输入联系内容"
          />
        </el-form-item>
        <el-form-item label="记录人" prop="operator">
          <el-input v-model="form.operator" placeholder="请输入记录人姓名" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitForm">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import { ref, reactive, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { contactApi, expertApi } from '../api/expert'

export default {
  name: 'ContactRecords',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const loading = ref(false)
    const records = ref([])
    const expertName = ref('')
    const dialogVisible = ref(false)
    const formRef = ref(null)
    
    const form = reactive({
      expertId: null,
      contactTime: new Date(),
      contactMethod: '',
      content: '',
      operator: ''
    })
    
    const rules = {
      contactTime: [{ required: true, message: '请选择联系时间', trigger: 'change' }],
      content: [{ required: true, message: '请输入联系内容', trigger: 'blur' }]
    }
    
    const fetchExpert = async () => {
      try {
        const res = await expertApi.getExpertById(route.params.id)
        if (res.data.success) {
          expertName.value = res.data.data.name
        }
      } catch (error) {
        ElMessage.error('获取专家信息失败')
      }
    }
    
    const fetchRecords = async () => {
      loading.value = true
      try {
        const res = await contactApi.getContactRecords(route.params.id)
        if (res.data.success) {
          records.value = res.data.data
        }
      } catch (error) {
        ElMessage.error('获取联系记录失败')
      } finally {
        loading.value = false
      }
    }
    
    const goBack = () => {
      router.push(`/expert/${route.params.id}`)
    }
    
    const showAddDialog = () => {
      form.expertId = parseInt(route.params.id)
      form.contactTime = new Date()
      form.contactMethod = ''
      form.content = ''
      form.operator = ''
      dialogVisible.value = true
    }
    
    const submitForm = async () => {
      const valid = await formRef.value.validate().catch(() => false)
      if (!valid) return
      
      try {
        await contactApi.addContactRecord(form)
        ElMessage.success('添加成功')
        dialogVisible.value = false
        fetchRecords()
        fetchExpert() // Refresh expert status
      } catch (error) {
        ElMessage.error(error.response?.data?.message || '添加失败')
      }
    }
    
    const getTimelineType = (method) => {
      const types = {
        '电话': 'primary',
        '邮件': 'success',
        '微信': 'warning',
        '面谈': 'danger'
      }
      return types[method] || 'info'
    }
    
    const formatDate = (date) => {
      if (!date) return '-'
      return new Date(date).toLocaleString('zh-CN')
    }
    
    onMounted(() => {
      fetchExpert()
      fetchRecords()
    })
    
    return {
      loading,
      records,
      expertName,
      dialogVisible,
      form,
      formRef,
      rules,
      goBack,
      showAddDialog,
      submitForm,
      getTimelineType,
      formatDate
    }
  }
}
</script>

<style scoped>
.contact-records {
  max-width: 1000px;
  margin: 0 auto;
}

.records-card {
  margin-top: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.record-card {
  margin-bottom: 10px;
}

.record-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.operator {
  font-size: 12px;
  color: #909399;
}
</style>
